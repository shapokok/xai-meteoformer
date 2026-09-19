"""Input-window sweep for the headline model (P0-2, Reviewer 1 Major #6).

seq_len is in neither the resume key nor the checkpoint tag, so each window
length lives in its own csv and directory:

    analysis/results_seqlen{L}.csv
    checkpoints/seqlen{L}/,  predictions/seqlen{L}/

L = 96 is the headline run and is read from analysis/results_clean.csv.

A different seq_len changes the number of test windows (the first window
starts L steps into the test split), but the windows are aligned at the END:
the last N forecast targets are identical for every L (checked against the
*_true.npy of each run). Diebold-Mariano against L = 96 is therefore run on
the common tail -- the windows present for both lengths -- and the
script asserts that the targets match there. Seeds: Welch t-test on MAE.

Outputs -> analysis/window_sweep.md
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import significance as sig  # noqa: E402


def fp(p):
    """p-value for a table; DM p underflows to 0 for large effects."""
    return "<1e-15" if p < 1e-15 else f"{p:.3g}"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL, ABL = "XAI-MeteoFormer", "no_revin"
LENGTHS = (24, 48, 96, 192)
DATASETS = [("jena", "Jena"), ("beijing_aotizhongxin", "Beijing (Aotizhongxin)")]


def load():
    parts = []
    for L in LENGTHS:
        if L == 96:
            d = pd.read_csv(os.path.join(ROOT, "analysis", "results_clean.csv"))
            d = d[(d.model == MODEL) & (d.ablation == ABL)].copy()
        else:
            f = os.path.join(ROOT, "analysis", f"results_seqlen{L}.csv")
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f)
        d["L"] = L
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    return d.drop_duplicates(["dataset", "L", "seed"], keep="last")


def pred_dir(L):
    base = os.path.join(ROOT, "predictions")
    return base if L == 96 else os.path.join(base, f"seqlen{L}")


def preds(ds, L):
    ps = []
    for s in range(5):
        f = os.path.join(pred_dir(L), f"{MODEL}_{ds}_{ABL}_s{s}_pred.npy")
        if os.path.exists(f):
            ps.append(np.load(f).astype(np.float64))
    return ps


def dm_vs_96(ds, L):
    """DM of window length L against 96 on the common tail of test windows."""
    tf = {l: os.path.join(pred_dir(l), f"{ds}_true.npy") for l in (L, 96)}
    if not all(os.path.exists(f) for f in tf.values()):
        return None
    t_l, t_96 = np.load(tf[L]), np.load(tf[96])
    n = min(len(t_l), len(t_96))
    assert np.allclose(t_l[-n:], t_96[-n:], atol=1e-5), f"{ds} L={L}: targets not aligned"
    pl, p96 = preds(ds, L), preds(ds, 96)
    if not pl or not p96:
        return None
    true = t_96[-n:].astype(np.float64)
    out = {"n": n}
    for kind in ("abs", "sq"):
        md_, _, p, _ = sig.diebold_mariano(
            sig.window_loss([p[-n:] for p in pl], true, kind),
            sig.window_loss([p[-n:] for p in p96], true, kind))
        out[kind] = (md_, p)
    return out


def main():
    d = load()
    L_ = []
    W = L_.append
    W("# Input-window sweep (P0-2)\n")
    W("Produced by [analysis/window_sweep.py](analysis/window_sweep.py). Headline "
      "model (no RevIN), everything else fixed, prediction length 24 h. "
      "L = 96 is the published configuration. Mean ± sd over seeds; p (Welch) "
      "is Welch's t-test on seed MAE against L = 96. DM is Diebold–Mariano "
      "against L = 96 on the test windows common to both lengths (the windows "
      "are aligned at the end of the test split; the script asserts the targets "
      "match), per-window loss averaged over seeds, Holm over the three "
      "lengths within each dataset and loss. Negative Δ favours L.\n")
    missing = []
    for ds, title in DATASETS:
        s = d[d.dataset == ds]
        ref = s[s.L == 96]
        W(f"## {title}\n")
        dms = {L: dm_vs_96(ds, L) for L in LENGTHS if L != 96}
        adj = {}
        for kind in ("abs", "sq"):
            ls = [L for L in dms if dms[L]]
            for L, pa in zip(ls, sig.holm([dms[L][kind][1] for L in ls])):
                adj[(L, kind)] = pa
        W("| L (h) | n seeds | val loss | MAE | ΔMAE vs 96 | p (Welch) | DM ΔL1 (p_Holm) | DM ΔL2 (p_Holm) | RMSE | R² |")
        W("|---|---|---|---|---|---|---|---|---|---|")
        for L in LENGTHS:
            g = s[s.L == L]
            if g.empty:
                missing.append(f"{ds} L={L}")
                W(f"| {L} | 0 | | _not run_ | | | | | | |")
                continue
            if L == 96 or ref.empty:
                dm, p, c1, c2 = "—", "—", "—", "—"
            else:
                dm = f"{g.MAE.mean() - ref.MAE.mean():+.3f}"
                p = f"{stats.ttest_ind(g.MAE, ref.MAE, equal_var=False).pvalue:.3g}"
                cs = []
                for kind in ("abs", "sq"):
                    if dms.get(L):
                        pa = adj[(L, kind)]
                        star = "\\*" if pa < 0.05 else ""
                        cs.append(f"{dms[L][kind][0]:+.4f}{star} ({fp(pa)}, n={dms[L]['n']})")
                    else:
                        cs.append("_predictions missing_")
                c1, c2 = cs
            W(f"| {L}{' (published)' if L == 96 else ''} | {g.seed.nunique()} | "
              f"{g.best_val.mean():.4f} | {g.MAE.mean():.3f} ± {g.MAE.std():.3f} | {dm} | {p} | {c1} | {c2} | "
              f"{g.RMSE.mean():.3f} ± {g.RMSE.std():.3f} | "
              f"{g.R2.mean():.3f} ± {g.R2.std():.3f} |")
            if g.seed.nunique() < 5:
                missing.append(f"{ds} L={L}: {g.seed.nunique()}/5 seeds")
        W("")
    W("## Selection on validation\n")
    for ds, title in DATASETS:
        v = d[d.dataset == ds].groupby("L").best_val.mean()
        if len(v):
            W(f"- {title}: lowest mean validation loss at **L = {int(v.idxmin())}** "
              f"({v.min():.4f}; L = 96: {v.get(96, float('nan')):.4f}).")
    W("")
    W("## Reading\n")
    W("Negative ΔMAE means the shorter/longer window is better than the published "
      "96 h. Consistent with [analysis/occlusion_time.md](analysis/occlusion_time.md), "
      "which finds the model relies on the last ~16 h of the window.\n")
    if missing:
        W("## Missing\n")
        for m in missing:
            W(f"- {m}")
        W("")
    dst = os.path.join(ROOT, "analysis", "window_sweep.md")
    open(dst, "w").write("\n".join(L_))
    print("->", dst, "| missing:", missing or "none")


if __name__ == "__main__":
    main()
