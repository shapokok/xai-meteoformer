"""Build both datasets into one common format.

Output per dataset, written to --out:
    {name}_X.npy      float32 (T, N)   hourly, gap-filled
    {name}_meta.json  column names, target positions, timestamps, freq

Both datasets end up with the SAME four targets — temperature, relative
humidity, pressure, wind speed — so the cross-dataset section compares
like with like. Beijing has no RH column, so it is derived from
temperature and dew point with the Magnus formula.

Run once, then upload the output folder as a Kaggle Dataset. Re-running
preprocessing inside every training notebook wastes GPU quota.

    python src/data/prepare.py --dataset jena     --out data/processed
    python src/data/prepare.py --dataset beijing  --out data/processed
"""

import argparse
import json
import os
import zipfile
from typing import Dict

import numpy as np
import pandas as pd

JENA_URL = ("https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
            "jena_climate_2009_2016.csv.zip")

# Beijing wind direction is categorical; map to degrees.
WD_MAP = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
    "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225,
    "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}

TARGETS = ["T", "RH", "P", "WS"]   # canonical target names in every dataset

# Channels present in BOTH datasets. Needed for the zero-shot transfer
# experiment: a model trained on Jena can only be evaluated on Beijing
# if the input tensor has identical columns in identical order.
COMMON_CHANNELS = TARGETS + ["Tdew", "wx", "wy",
                             "hour_sin", "hour_cos", "doy_sin", "doy_cos"]


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    hour = idx.hour.values.astype(np.float32)
    doy = idx.dayofyear.values.astype(np.float32)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def prepare_jena(raw_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(raw_dir, "jena_climate_2009_2016.csv")
    if not os.path.exists(csv_path):
        zip_path = csv_path + ".zip"
        if not os.path.exists(zip_path):
            import urllib.request
            os.makedirs(raw_dir, exist_ok=True)
            print(f"downloading {JENA_URL}")
            urllib.request.urlretrieve(JENA_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)

    df = pd.read_csv(csv_path)
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.set_index("Date Time").sort_index()

    # Known data fault: wind columns use -9999 as a missing sentinel.
    for c in ["wv (m/s)", "max. wv (m/s)"]:
        df.loc[df[c] < -9000.0, c] = 0.0

    # Wind direction -> vector components before averaging, otherwise the
    # hourly mean of 359 deg and 1 deg comes out as 180 deg.
    rad = np.deg2rad(df["wd (deg)"].values)
    df["wx"] = df["wv (m/s)"].values * np.cos(rad)
    df["wy"] = df["wv (m/s)"].values * np.sin(rad)
    df = df.drop(columns=["wd (deg)"])

    df = df.resample("1h").mean()
    df = df.interpolate(limit_direction="both")

    df = df.rename(columns={
        "T (degC)": "T", "rh (%)": "RH", "p (mbar)": "P", "wv (m/s)": "WS",
        "Tpot (K)": "Tpot", "Tdew (degC)": "Tdew", "VPmax (mbar)": "VPmax",
        "VPact (mbar)": "VPact", "VPdef (mbar)": "VPdef", "sh (g/kg)": "SH",
        "H2OC (mmol/mol)": "H2OC", "rho (g/m**3)": "rho",
        "max. wv (m/s)": "WSmax",
    })
    return _add_time_features(df)


def _find_station_csvs(raw_dir: str) -> Dict[str, str]:
    """Map station name -> csv path for every PRSA_Data_*.csv under raw_dir."""
    found: Dict[str, str] = {}
    for root, _dirs, files in os.walk(raw_dir):
        for f in files:
            if f.startswith("PRSA_Data_") and f.endswith(".csv"):
                # PRSA_Data_<Station>_20130301-20170228.csv
                parts = f[len("PRSA_Data_"):].rsplit("_", 1)
                if parts:
                    found[parts[0]] = os.path.join(root, f)
    return found


def _zip_is_relevant(path: str) -> bool:
    """Peek inside without extracting.

    Only archives that hold PRSA station CSVs — or another zip that
    might — are our business. Without this check the walker also tries
    to unpack the Jena archive sitting in the same folder and collides
    with the CSV already extracted next to it.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except (zipfile.BadZipFile, OSError):
        return False
    has_station = any(
        os.path.basename(n).startswith("PRSA_Data_") and n.endswith(".csv")
        for n in names
    )
    has_nested_zip = any(n.lower().endswith(".zip") for n in names)
    return has_station or has_nested_zip


def _extract_nested_zips(raw_dir: str, max_depth: int = 3) -> None:
    """UCI ships this dataset as a zip inside a zip.

    Unpack only the relevant archives, each into a sibling folder, until
    the station CSVs appear. Already-extracted archives are skipped, so
    this is cheap and safe to re-run.
    """
    for _ in range(max_depth):
        if _find_station_csvs(raw_dir):
            return
        zips = [os.path.join(r, f)
                for r, _d, fs in os.walk(raw_dir)
                for f in fs if f.lower().endswith(".zip")]
        did_work = False
        for z in zips:
            if not _zip_is_relevant(z):
                continue
            stem = os.path.splitext(z)[0]
            # If the archive was already unpacked next to itself, leave it.
            if os.path.isdir(stem) or os.path.isdir(stem + "__extracted"):
                continue
            target = stem if not os.path.exists(stem) else stem + "__extracted"
            print(f"extracting {os.path.basename(z)}")
            os.makedirs(target, exist_ok=True)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(target)
            did_work = True
        if not did_work:
            return


def prepare_beijing(raw_dir: str, station: str = "Aotizhongxin") -> pd.DataFrame:
    """Load one station of the UCI Beijing Multi-Site Air-Quality dataset.

    Point --raw at whatever folder the UCI download landed in; nested
    zips are unpacked automatically. On Kaggle, add the public dataset as
    a notebook input and pass its path instead of downloading.
    """
    _extract_nested_zips(raw_dir)
    stations = _find_station_csvs(raw_dir)
    if not stations:
        raise FileNotFoundError(
            f"no PRSA_Data_*.csv found under '{raw_dir}'. Download "
            "'Beijing Multi-Site Air-Quality Data' from UCI and point "
            "--raw at the folder containing the archive."
        )
    if station not in stations:
        raise FileNotFoundError(
            f"station '{station}' not found. Available: "
            f"{', '.join(sorted(stations))}"
        )
    hit = stations[station]
    print(f"using {hit}")
    df = pd.read_csv(hit)
    df["Date Time"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.set_index("Date Time").sort_index()
    df = df.drop(columns=[c for c in ["No", "year", "month", "day", "hour",
                                      "station"] if c in df.columns])

    deg = df["wd"].map(WD_MAP)
    rad = np.deg2rad(deg.values.astype(np.float64))
    df["wx"] = df["WSPM"].values * np.cos(rad)
    df["wy"] = df["WSPM"].values * np.sin(rad)
    df = df.drop(columns=["wd"])

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.interpolate(limit_direction="both")

    # Relative humidity from temperature and dew point (Magnus, Alduchov
    # & Eskridge 1996 coefficients). Gives Beijing the same target set
    # as Jena.
    t, td = df["TEMP"].values, df["DEWP"].values
    a, b = 17.625, 243.04
    rh = 100.0 * np.exp(a * td / (b + td)) / np.exp(a * t / (b + t))
    df["RH"] = np.clip(rh, 0.0, 100.0)

    df = df.rename(columns={"TEMP": "T", "PRES": "P", "WSPM": "WS",
                            "DEWP": "Tdew", "RAIN": "rain"})
    return _add_time_features(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["jena", "beijing"], required=True)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--station", default="Aotizhongxin")
    ap.add_argument("--common_channels", action="store_true",
                    help="keep only the 11 channels shared by both datasets, "
                         "for zero-shot cross-dataset transfer")
    args = ap.parse_args()

    if args.dataset == "jena":
        df = prepare_jena(args.raw)
        name = "jena"
    else:
        df = prepare_beijing(args.raw, args.station)
        name = f"beijing_{args.station.lower()}"

    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        raise RuntimeError(f"missing targets {missing}; got {list(df.columns)}")

    # targets first, then the rest — keeps target_idx stable at [0,1,2,3]
    if args.common_channels:
        missing_c = [c for c in COMMON_CHANNELS if c not in df.columns]
        if missing_c:
            raise RuntimeError(f"common channels absent: {missing_c}")
        cols = list(COMMON_CHANNELS)          # fixed order across datasets
        name = f"{name}_common"
    else:
        cols = TARGETS + [c for c in df.columns if c not in TARGETS]
    df = df[cols].astype(np.float32)

    assert not df.isna().any().any(), "NaNs survived interpolation"
    assert np.isfinite(df.values).all(), "non-finite values in output"

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, f"{name}_X.npy"), df.values)
    meta = {
        "name": name,
        "columns": cols,
        "target_idx": [cols.index(t) for t in TARGETS],
        "target_names": TARGETS,
        "n_rows": int(df.shape[0]),
        "n_channels": int(df.shape[1]),
        "freq": "1h",
        "common_channels": bool(args.common_channels),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
    }
    with open(os.path.join(args.out, f"{name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
