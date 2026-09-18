"""Variance calibration fitted on validation, applied to test.

For every (model, ablation, seed, target channel) a single scalar `a` is
fitted on the VALIDATION split by least squares:

    pred_cal = m + a * (pred - m),     m = mean of the validation predictions

`a` minimises validation MSE in closed form. Both `a` and `m` come from
validation only; the test set is then transformed with those frozen
numbers and scored once. Nothing is fitted, selected or thresholded on
test.

`a < 1` shrinks the forecast toward its own mean -- the variance
shrinkage a squared loss rewards. `a > 1` would inflate it.

Outputs -> analysis/calibration_metrics.csv  (per model/seed, pre and post)
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL = os.path.join(ROOT, "predictions_val")
TEST = os.path.join(ROOT, "predictions")
NAMES = ["T", "RH", "P", "WS"]
DATASETS = ["jena", "beijing_aotizhongxin"]


def fit_a(xv, yv):
    """closed-form least-squares scale about the validation mean"""
    m = xv.mean()
    num = ((yv - m) * (xv - m)).sum()
    den = ((xv - m) ** 2).sum()
    return m, float(num / den) if den > 0 else 1.0


def apply_cal(x, m, a):
    return m + a * (x - m)


def metrics(pred, true):
    """pred/true (N,H,C) -> dict of aggregate + per-channel MAE/RMSE/R2"""
    out = {}
    e = pred - true
    out["MAE"] = float(np.abs(e).mean())
    out["RMSE"] = float(np.sqrt((e ** 2).mean()))
    ss_res = (e ** 2).sum(axis=(0, 1))
    ss_tot = ((true - true.mean(axis=(0, 1), keepdims=True)) ** 2).sum(axis=(0, 1))
    out["R2"] = float(np.mean(1.0 - ss_res / np.clip(ss_tot, 1e-9, None)))
    for c, n in enumerate(NAMES):
        ec = e[:, :, c]
        out[f"MAE_{n}"] = float(np.abs(ec).mean())
        out[f"RMSE_{n}"] = float(np.sqrt((ec ** 2).mean()))
        out[f"R2_{n}"] = float(1.0 - ss_res[c] / max(ss_tot[c], 1e-9))
    return out


def main():
    rows = []
    percase = {}
    for ds in DATASETS:
        vt_f = os.path.join(VAL, f"{ds}_val_true.npy")
        tt_f = os.path.join(TEST, f"{ds}_true.npy")
        if not os.path.exists(vt_f):
            print(f"!! missing {vt_f} -- skipping {ds}")
            continue
        vtrue = np.load(vt_f).astype(np.float64)
        ttrue = np.load(tt_f).astype(np.float64)

        for f in sorted(os.listdir(VAL)):
            if not f.endswith("_valpred.npy") or f"_{ds}_" not in f:
                continue
            tag = f[: -len("_valpred.npy")]
            tf = os.path.join(TEST, f"{tag}_pred.npy")
            if not os.path.exists(tf):
                print(f"!! no test prediction for {tag}")
                continue
            vp = np.load(os.path.join(VAL, f)).astype(np.float64)
            tp = np.load(tf).astype(np.float64)
            assert vp.shape[0] == vtrue.shape[0] and tp.shape[0] == ttrue.shape[0]

            cal = tp.copy()
            a_by_ch = {}
            for c in range(len(NAMES)):
                m, a = fit_a(vp[:, :, c], vtrue[:, :, c])
                cal[:, :, c] = apply_cal(tp[:, :, c], m, a)
                a_by_ch[NAMES[c]] = a

            model, rest = tag.split(f"_{ds}_")
            ablation, seed = rest.rsplit("_s", 1)
            pre, post = metrics(tp, ttrue), metrics(cal, ttrue)
            row = {"dataset": ds, "model": model, "ablation": ablation,
                   "seed": int(seed)}
            row.update({f"a_{k}": v for k, v in a_by_ch.items()})
            row.update({f"pre_{k}": v for k, v in pre.items()})
            row.update({f"post_{k}": v for k, v in post.items()})
            rows.append(row)
            percase[(ds, model, ablation, int(seed))] = (tp, cal, ttrue)

    df = pd.DataFrame(rows)
    dst = os.path.join(ROOT, "analysis", "calibration_metrics.csv")
    df.to_csv(dst, index=False)
    print(f"{len(df)} runs calibrated -> {dst}")
    json.dump({"n_runs": len(df)},
              open(os.path.join(ROOT, "analysis", "_cal_state.json"), "w"))
    return df, percase


if __name__ == "__main__":
    main()
