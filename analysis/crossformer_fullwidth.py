"""Crossformer at full width vs the proposed model (P0-1).

MODEL_OVERRIDES capped Crossformer at d_model=d_ff=128 with a comment
about 8 GB of VRAM, but training ran on a 16 GB T4. This compares the
three capacity points once the Kaggle runs are back:

    128 x  128   1.70 M params   (what the paper reports)
    256 x  512   7.98 M params
    256 x 1024  10.61 M params

against the proposed model at 1.89 M. Parameter counts are reported in
the table because at 256 wide Crossformer is 4-6x our size, so a win
there is a statement about capacity, not architecture.

Adds Diebold-Mariano with Holm, reusing analysis/significance.py.

Run after analysis/repair_results_csv.py has picked up results_new*.csv.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
PRED = os.path.join(ROOT, "predictions")
OURS = ("XAI-MeteoFormer", "no_revin", "default")
DS = "jena"
SEEDS = range(5)


def load(model, ablation, width):
    """per-seed test predictions for one configuration"""
    sfx = "" if width == "default" else f"_w{width}"
    out = []
    for s in SEEDS:
        f = os.path.join(PRED, f"{model}_{DS}_{ablation}{sfx}_s{s}_pred.npy")
        if os.path.exists(f):
            out.append(np.load(f).astype(np.float64))
    return out


def main():
    d = pd.read_csv(os.path.join(ROOT, "analysis", "results_clean.csv"))
    d = d[d.dataset == DS]
    if "width" not in d.columns:
        d["width"] = "default"

    configs = [("Crossformer", "full", "default", "Crossformer 128x128"),
               ("Crossformer", "full", "256x512", "Crossformer 256x512"),
               ("Crossformer", "full", "256x1024", "Crossformer 256x1024"),
               (*OURS, "MeteoFormer (ours)")]

    L = []
    W = L.append
    W("# Crossformer at full width (P0-1)\n")
    W("Produced by [analysis/crossformer_fullwidth.py](analysis/crossformer_fullwidth.py). "
      "Jena, 5 seeds, mean ± sd. The normalization configuration is unchanged "
      "(none, as in the published runs).\n")
    W("`MODEL_OVERRIDES` capped Crossformer at `d_model=d_ff=128` with a comment "
      "about 8 GB of VRAM, but training ran on a 16 GB T4, so the cap was "
      "inherited rather than required. Measured peak memory at 256x1024, "
      "batch 64 is in `analysis/crossformer_probe.json`.\n")

    W("## Accuracy by capacity\n")
    W("| Configuration | params | n seeds | MAE | RMSE | R² |")
    W("|---|---|---|---|---|---|")
    missing = []
    for model, abl, width, label in configs:
        g = d[(d.model == model) & (d.ablation == abl) & (d.width == width)]
        if g.empty:
            missing.append(label)
            W(f"| {label} | — | 0 | _not run yet_ | — | — |")
            continue
        p = int(g.params.iloc[0]) if g.params.notna().any() else 0
        W(f"| {label} | {p:,} | {g.seed.nunique()} | "
          f"{g.MAE.mean():.3f} ± {g.MAE.std(ddof=1):.3f} | "
          f"{g.RMSE.mean():.3f} ± {g.RMSE.std(ddof=1):.3f} | "
          f"{g.R2.mean():.3f} ± {g.R2.std(ddof=1):.3f} |")
    W("")
    if missing:
        W(f"**Missing: {', '.join(missing)}** — run cells 6/7 of "
          "`notebooks/kaggle_resubmit.ipynb`, download, then re-run "
          "`analysis/repair_results_csv.py` and this script.\n")

    W("## Diebold–Mariano against the proposed model\n")
    ours = load(*OURS)
    if not ours:
        W("_proposed-model predictions not found_\n")
    else:
        import significance as sig
        true = np.load(os.path.join(PRED, f"{DS}_true.npy")).astype(np.float64)
        W("| Configuration | Δ L1 | p | Δ L2 | p |")
        W("|---|---|---|---|---|")
        rows, praw = [], {"abs": [], "sq": []}
        for model, abl, width, label in configs[:-1]:
            ps = load(model, abl, width)
            if not ps:
                rows.append((label, None))
                continue
            cell = {}
            for kind in ("abs", "sq"):
                la = sig.window_loss(ours, true, kind)
                lb = sig.window_loss(ps, true, kind)
                md_, st, p, _ = sig.diebold_mariano(la, lb)
                cell[kind] = (md_, p)
                praw[kind].append(p)
            rows.append((label, cell))
        adj = {k: sig.holm(v) for k, v in praw.items() if v}
        i = 0
        for label, cell in rows:
            if cell is None:
                W(f"| {label} | _not run yet_ | | | |")
                continue
            cs = []
            for kind in ("abs", "sq"):
                md_, _ = cell[kind]
                pa = adj[kind][i]
                cs += [f"{md_:+.4f}{'*' if pa < 0.05 else ''}",
                       f"{pa:.3g}"]
            W(f"| {label} | " + " | ".join(cs) + " |")
            i += 1
        W("")
        W("Negative Δ favours the proposed model. `*` = p<0.05 after Holm "
          "within each column. L1 = absolute loss, L2 = squared loss; the "
          "per-window loss is averaged over seeds, not the predictions.\n")

    W("## How to read this\n")
    W("If Crossformer improves with width, the published comparison "
      "under-provisioned it and the paper must say so — reporting the capped "
      "configuration without that caveat would not survive review. If it does "
      "not improve, the cap was harmless and the existing numbers stand, which "
      "is worth one sentence in the setup section.\n")
    W("Either way the **parameter column stays in the table**: at 256 wide "
      "Crossformer is 4–6× the proposed model, so a win at that size is a "
      "statement about capacity, not about architecture.\n")

    dst = os.path.join(ROOT, "analysis", "crossformer_fullwidth.md")
    open(dst, "w").write("\n".join(L))
    print("->", dst)


if __name__ == "__main__":
    main()
