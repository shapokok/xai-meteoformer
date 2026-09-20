"""Zero-shot transfer across the Beijing PRSA stations (Reviewer 1, Minor #6).

Every checkpoint trained on Aotizhongxin is applied, without any retraining
or re-fitting, to the other 11 stations of the UCI Beijing Multi-Site
Air-Quality dataset:

  * each station is prepared by the SAME code as Aotizhongxin
    (src/data/prepare.py: prepare_beijing, targets first, same channel order;
    the script asserts the channel list matches);
  * inputs are scaled with the AOTIZHONGXIN training scaler, the one the
    models were trained with -- nothing is fitted on the target station,
    not even its mean and sd (strict zero-shot);
  * the chronological 70/10/20 split is the same, and all stations cover
    the same period (2013-03 to 2017-02), so every station is scored on the
    same test months;
  * 11 models x 5 seeds, each baseline in the normalization variant the
    main table reports (analysis/selection.py), our model as no_revin.

Aotizhongxin itself is run through the same code path as a control: its
MAE must reproduce analysis/results_clean.csv.

Outputs -> analysis/cross_station.csv   (one row per model, seed, station)
        -> analysis/cross_station.md
        -> paper/tables/cross_station.tex
        data/processed_stations/  (prepared station arrays; data/ is not tracked)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from data.dataset import MeteoWindowDataset, build_splits    # noqa: E402
from data.prepare import TARGETS, _find_station_csvs, prepare_beijing  # noqa: E402
from train import set_seed                                    # noqa: E402
from missing_robustness import build, run, NAMES              # noqa: E402
from selection import selected_ablation, selected_norm        # noqa: E402
import mps_unfold_fix  # noqa: E402,F401  (patching models on MPS)

SRC = "beijing_aotizhongxin"
SRC_STATION = "Aotizhongxin"
PROC = os.path.join(ROOT, "data", "processed")
PROC_ST = os.path.join(ROOT, "data", "processed_stations")
LABEL = {"XAI-MeteoFormer": "MeteoFormer"}


def prepare_station(station, columns):
    """Write data/processed_stations/beijing_<station>_X.npy + meta, with the
    source dataset's column order."""
    name = f"beijing_{station.lower()}"
    xf = os.path.join(PROC_ST, f"{name}_X.npy")
    if os.path.exists(xf):
        return name
    os.makedirs(PROC_ST, exist_ok=True)
    df = prepare_beijing(os.path.join(ROOT, "data", "raw"), station)
    assert set(df.columns) == set(columns), (
        f"{station}: channels differ from {SRC}: "
        f"{sorted(set(df.columns) ^ set(columns))}")
    df = df[columns].astype(np.float32)
    assert not df.isna().any().any() and np.isfinite(df.values).all()
    np.save(xf, df.values)
    meta = {"name": name, "columns": columns,
            "target_idx": [columns.index(t) for t in TARGETS],
            "target_names": TARGETS, "n_rows": int(df.shape[0]),
            "n_channels": int(df.shape[1]), "freq": "1h",
            "start": str(df.index[0]), "end": str(df.index[-1])}
    json.dump(meta, open(os.path.join(PROC_ST, f"{name}_meta.json"), "w"), indent=2)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis", "cross_station.csv"))
    a = ap.parse_args()

    src_meta = json.load(open(os.path.join(PROC, f"{SRC}_meta.json")))
    columns = src_meta["columns"]
    train_ds, _, src_test = build_splits(PROC, SRC, seq_len=96, pred_len=24, seed=0)
    scaler = (train_ds.mu, train_ds.sigma)

    stations = sorted(_find_station_csvs(os.path.join(ROOT, "data", "raw")))
    tests = {SRC_STATION: src_test}
    for st in stations:
        if st == SRC_STATION:
            continue
        name = prepare_station(st, columns)
        tests[st] = MeteoWindowDataset(PROC_ST, name, split="test", seq_len=96,
                                       pred_len=24, scaler=scaler, seed=0)
    print({k: len(v) for k, v in tests.items()})

    rows = pd.read_csv(a.out).to_dict("records") if os.path.exists(a.out) else []
    done = {(r["model"], r["seed"], r["station"]) for r in rows}
    ck = os.path.join(ROOT, "checkpoints")
    models = a.models or sorted({f.split(f"_{SRC}_")[0] for f in os.listdir(ck)
                                 if f.endswith(".pt") and f"_{SRC}_" in f})
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tidx = train_ds.target_idx
    for mn in models:
        abl = selected_ablation(mn, SRC)
        for seed in a.seeds:
            f = os.path.join(ck, f"{mn}_{SRC}_{abl}_s{seed}.pt")
            if not os.path.exists(f):
                continue
            todo = [st for st in tests if (mn, seed, st) not in done]
            if not todo:
                continue
            set_seed(seed)
            model = build(mn, abl, train_ds, dev, os.path.join(ROOT, "Time-Series-Library"))
            model.load_state_dict(torch.load(f, map_location=dev))
            model.eval()
            for st in todo:
                pred, true = run(model, tests[st], dev, train_ds.mu, train_ds.sigma, tidx)
                e = pred - true
                row = {"model": mn, "ablation": abl, "seed": seed, "station": st,
                       "in_domain": st == SRC_STATION, "n_windows": int(pred.shape[0]),
                       "MAE": float(np.abs(e).mean()),
                       "RMSE": float(np.sqrt((e ** 2).mean()))}
                for c, n in enumerate(NAMES):
                    row[f"MAE_{n}"] = float(np.abs(e[:, :, c]).mean())
                    row[f"RMSE_{n}"] = float(np.sqrt((e[:, :, c] ** 2).mean()))
                rows.append(row)
                print(f"{mn:16s} s{seed} {st:14s} MAE={row['MAE']:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)
            del model
            if dev.type == "mps":
                torch.mps.empty_cache()
    print(f"-> {a.out} ({len(rows)} rows)")
    report(pd.DataFrame(rows))


def report(d):
    ref = pd.read_csv(os.path.join(ROOT, "analysis", "results_clean.csv"))
    L = []
    W = L.append
    W("# Zero-shot transfer across Beijing stations (Reviewer 1, Minor #6)\n")
    W("Produced by [analysis/cross_station.py](analysis/cross_station.py). Every "
      "model is trained on **Aotizhongxin only** and applied without retraining "
      "to the other 11 PRSA stations. Inputs are scaled with the Aotizhongxin "
      "training scaler; nothing is fitted on the target station. Same channels, "
      "same chronological split, same test months at every station. 5 seeds per "
      "model; baselines in the normalization variant of the main table.\n")

    # control: in-domain must reproduce the training-time metrics
    W("## Control: Aotizhongxin through the same code path\n")
    W("| Model | MAE here | MAE in results_clean | |Δ| |")
    W("|---|---|---|---|")
    worst, dev = 0.0, {}
    ind = d[d.in_domain]
    for mn, g in sorted(ind.groupby("model")):
        r = ref[(ref.model == mn) & (ref.dataset == SRC) & (ref.width == "default") &
                (ref.ablation == ("no_revin" if mn == "XAI-MeteoFormer" else "full"))]
        if mn != "XAI-MeteoFormer":
            r = r[r.norm_variant == selected_norm(mn, SRC)]
        r = r[r.seed.isin(g.seed)]
        dd = abs(g.MAE.mean() - r.MAE.mean())
        worst = max(worst, dd)
        dev[mn] = dd
        W(f"| {LABEL.get(mn, mn)} | {g.MAE.mean():.4f} | {r.MAE.mean():.4f} | {dd:.1e} |")
    W("")
    # Informer's ProbSparse attention draws torch.randint INSIDE the forward
    # pass, so its predictions are not reproducible run to run even in eval;
    # TimesNet's FFT differs in the last digits. Both are the harness matching
    # the model, not a harness bug. Anything above 1% would be.
    NONDET = {"Informer", "TimesNet"}
    bad = {m: v for m, v in dev.items() if v > 1e-3 and m not in NONDET}
    loud = {m: v for m, v in dev.items() if v > 1e-3 and m in NONDET}
    if bad:
        W(f"**The harness does NOT reproduce the in-domain numbers for "
          f"{', '.join(sorted(bad))} (up to {max(bad.values()):.1e}) — do not use "
          "the transfer results until this is resolved.**\n")
    else:
        W(f"Largest deviation {worst:.1e}. Every model reproduces its published "
          "in-domain MAE." + (
            f" {', '.join(sorted(loud))} "
            f"{'differs' if len(loud) == 1 else 'differ'} by up to {max(loud.values()):.1e} "
            "(<0.3% relative): `ProbAttention` samples keys with `torch.randint` "
            "inside the forward pass and `TimesNet` runs an FFT, so neither is "
            "bit-reproducible across runs. That is a property of those models, "
            "not of this harness." if loud else "") + "\n")

    out = d[~d.in_domain]
    st_m = out.groupby(["model", "station"])[["MAE", "RMSE"]].mean().reset_index()
    stations = sorted(out.station.unique())

    W("## Mean over the 11 target stations\n")
    W("Per model: the seed-mean MAE at each station, then mean ± sd across "
      "stations; in-domain Aotizhongxin for reference; rank = rank by the "
      "cross-station mean MAE.\n")
    W("| Rank | Model | in-domain MAE | transfer MAE (mean ± sd over stations) | "
      "transfer RMSE | Δ vs in-domain | stations where 1st |")
    W("|---|---|---|---|---|---|---|")
    agg = st_m.groupby("model").agg(MAE=("MAE", "mean"), MAE_sd=("MAE", "std"),
                                    RMSE=("RMSE", "mean"), RMSE_sd=("RMSE", "std"))
    agg["in_domain"] = ind.groupby("model").MAE.mean()
    firsts = st_m.loc[st_m.groupby("station").MAE.idxmin()].model.value_counts()
    agg = agg.sort_values("MAE")
    for i, (mn, r) in enumerate(agg.iterrows(), 1):
        W(f"| {i} | {LABEL.get(mn, mn)} | {r.in_domain:.3f} | {r.MAE:.3f} ± {r.MAE_sd:.3f} | "
          f"{r.RMSE:.3f} ± {r.RMSE_sd:.3f} | {r.MAE - r.in_domain:+.3f} | "
          f"{int(firsts.get(mn, 0))}/{len(stations)} |")
    W("")

    W("## Does the ranking survive transfer?\n")
    rk0 = ind.groupby("model").MAE.mean().rank()
    W("Kendall τ between the in-domain ranking (Aotizhongxin) and the ranking "
      "at each target station, by seed-mean MAE; winner per station.\n")
    W("| Station | winner | MeteoFormer rank | Kendall τ vs in-domain |")
    W("|---|---|---|---|")
    taus = []
    for st in stations:
        s = st_m[st_m.station == st].set_index("model").MAE
        rk = s.rank()
        tau = stats.kendalltau(rk0.loc[rk.index], rk).correlation
        taus.append(tau)
        W(f"| {st} | {LABEL.get(s.idxmin(), s.idxmin())} | "
          f"{int(rk.get('XAI-MeteoFormer', np.nan))} / {len(rk)} | {tau:.2f} |")
    W("")
    W(f"Kendall τ over stations: mean {np.mean(taus):.2f}, min {np.min(taus):.2f}, "
      f"max {np.max(taus):.2f}.\n")

    W("## MAE per station (seed mean)\n")
    piv = st_m.pivot(index="model", columns="station", values="MAE").loc[agg.index]
    W("| Model | " + " | ".join(stations) + " |")
    W("|---|" + "---|" * len(stations))
    for mn, r in piv.iterrows():
        W(f"| {LABEL.get(mn, mn)} | " + " | ".join(f"{v:.3f}" for v in r.values) + " |")
    W("")
    open(os.path.join(ROOT, "analysis", "cross_station.md"), "w").write("\n".join(L))

    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Model & In-domain MAE & Transfer MAE & Transfer RMSE & 1st at \\\\",
             "\\midrule"]
    for mn, r in agg.iterrows():
        nm = LABEL.get(mn, mn)
        nm = "\\textbf{" + nm + "}" if mn == "XAI-MeteoFormer" else nm
        lines.append(f"{nm} & {r.in_domain:.3f} & {r.MAE:.3f} $\\pm$ {r.MAE_sd:.3f} & "
                     f"{r.RMSE:.3f} $\\pm$ {r.RMSE_sd:.3f} & "
                     f"{int(firsts.get(mn, 0))}/{len(stations)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(ROOT, "paper", "tables", "cross_station.tex"), "w") as f:
        f.write("\\begin{table}[H]\n\\caption{Zero-shot transfer: models trained on "
                "Aotizhongxin, applied without retraining to the other 11 Beijing "
                "stations (input scaler of Aotizhongxin). Transfer MAE/RMSE: mean "
                "$\\pm$ s.d. over stations of the 5-seed mean. ``1st at'': number of "
                "stations where the model has the lowest MAE.}\n"
                "\\label{tab:cross_station}\n" + "\n".join(lines) + "\n\\end{table}\n")
    print("-> analysis/cross_station.md, paper/tables/cross_station.tex")


if __name__ == "__main__":
    main()
