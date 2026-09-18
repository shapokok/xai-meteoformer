"""Repair the column-shift in the results CSVs and de-duplicate.

Two defects, both diagnosed in analysis/results_hygiene.md:

1. Rows written before `lambda_ent` moved from the metric tail into the
   resume key carry a one-position rotation across the block
   params..lambda_ent: the value sitting in `params` is really
   `lambda_ent`, and every other field in that block is one column to the
   right of where it belongs. Detectable without ambiguity because
   `seq_len` is 96 in every run ever made, and reads as the epoch count
   in a rotated row.
2. The same runs are logged twice, once in each layout.

The metric columns (MAE onward) are NOT affected -- verified against
predictions/*_pred.npy, which reproduce them to 5 decimals in both
layouts.

Originals are never written to; output goes to analysis/.
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCK = ["params", "train_time_s", "infer_time_s", "best_val", "epochs_run",
         "seq_len", "pred_len", "d_model", "n_layers", "lr", "lambda_ent"]
# Extended identity of a run. Older CSVs predate norm_variant/width; those
# rows are back-filled with the configuration they were actually trained in.
KEY = ["model", "dataset", "ablation", "seed", "missing_rate"]
EXTRA_DEFAULTS = {"norm_variant": None, "width": "default",
                  "bl_d_model": None, "bl_d_ff": None}


def repair(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[BLOCK] = df[BLOCK].astype(float)
    shifted = (df["seq_len"] != 96).values
    if shifted.any():
        # rotate the block one position left: params <- train_time_s, ...,
        # lr <- lambda_ent, and the displaced first value becomes lambda_ent
        vals = df.loc[shifted, BLOCK].values
        df.loc[shifted, BLOCK] = np.concatenate([vals[:, 1:], vals[:, :1]], axis=1)
    assert (df["seq_len"] == 96).all(), "repair failed: seq_len still off"
    return df


def backfill(df: pd.DataFrame) -> pd.DataFrame:
    """Give pre-flag rows the configuration they were actually trained in."""
    df = df.copy()
    for c, v in EXTRA_DEFAULTS.items():
        if c not in df.columns:
            df[c] = v
    # LSTM historically carried our RevIN; every other baseline had none;
    # the proposed model's normalization is an ablation, not this flag.
    hist = np.where(df["model"].eq("LSTM"), "on", "off")
    df["norm_variant"] = df["norm_variant"].fillna(pd.Series(hist, index=df.index))
    df["width"] = df["width"].fillna("default")
    return df


def main():
    import glob
    out = []
    files = ["jena_results.csv", "beijing_results.csv"]
    files += [os.path.relpath(p, ROOT) for p in
              sorted(glob.glob(os.path.join(ROOT, "results_new*.csv")))]
    for f in files:
        if not os.path.exists(os.path.join(ROOT, f)):
            continue
        raw = pd.read_csv(os.path.join(ROOT, f))
        n_shift = int((raw["seq_len"] != 96).sum())
        fixed = backfill(repair(raw))
        before = len(fixed)
        # exact duplicates of the same run; keep the first
        fixed = fixed.drop_duplicates(
            subset=KEY + ["lambda_ent", "norm_variant", "width"], keep="first")
        dropped = before - len(fixed)
        print(f"{f}: {before} rows, {n_shift} column-shifted, "
              f"{dropped} duplicate runs dropped -> {len(fixed)}")
        out.append(fixed)

    clean = pd.concat(out, ignore_index=True)
    clean = clean.drop_duplicates(
        subset=KEY + ["lambda_ent", "norm_variant", "width"], keep="first")
    dst = os.path.join(ROOT, "analysis", "results_clean.csv")
    clean.to_csv(dst, index=False)
    print(f"-> {dst}  ({len(clean)} rows)")

    # sanity: one row per (model, dataset, ablation, seed)
    dup = clean.duplicated(subset=KEY + ["norm_variant", "width"]).sum()
    print("remaining duplicate keys:", dup)
    print("\nruns per group:")
    print(clean.groupby(["dataset", "model", "ablation", "width"])
          .size().to_string())


if __name__ == "__main__":
    main()
