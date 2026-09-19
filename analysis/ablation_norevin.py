"""Component ablations from the headline configuration (no_revin).

Each row removes one component on top of use_revin=False, 5 seeds, both
datasets. The older ablations from `full` (RevIN on) are a different
configuration and stay in paper/tables/ablation.tex as the appendix table.

Removing var_attn makes the attention constant, so the entropy term has no
gradient path: that row removes BOTH components, and says so.

Adds Diebold-Mariano against the headline model, reusing
analysis/significance.py (per-window loss averaged over seeds, HAC variance,
HLN correction, Holm within each dataset x loss family).

Run after analysis/repair_results_csv.py.
Outputs -> analysis/ablation_norevin.md
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import significance as sig  # noqa: E402


def fp(p):
    """p-value for a table; DM p underflows to 0 for large effects."""
    return "<1e-15" if p < 1e-15 else f"{p:.3g}"

PRED = os.path.join(ROOT, "predictions")
MODEL = "XAI-MeteoFormer"
BASE = "no_revin"
ROWS = [("no_revin+no_multiscale", "− multi-scale patching"),
        ("no_revin+no_var_attn", "− variable attention **and entropy term** (joint)"),
        ("no_revin+no_temp_attn", "− temporal attention"),
        ("no_revin+no_fusion", "− fusion gate"),
        ("no_revin+no_entropy", "− entropy term"),
        ("no_revin+no_cls", "− event head")]
DATASETS = [("jena", "Jena"), ("beijing_aotizhongxin", "Beijing (Aotizhongxin)")]


def load(ds, abl):
    ps = []
    for s in range(5):
        f = os.path.join(PRED, f"{MODEL}_{ds}_{abl}_s{s}_pred.npy")
        if os.path.exists(f):
            ps.append(np.load(f).astype(np.float64))
    return ps


def main():
    d = pd.read_csv(os.path.join(ROOT, "analysis", "results_clean.csv"))
    d = d[(d.model == MODEL)]
    L = []
    W = L.append
    W("# Ablation from the headline configuration (no RevIN)\n")
    W("Produced by [analysis/ablation_norevin.py](analysis/ablation_norevin.py) "
      "from [analysis/results_clean.csv](analysis/results_clean.csv). Each row "
      "removes one component from the published model (`no_revin`); mean ± sd "
      "over seeds. ΔMAE is relative to the headline model: **positive means "
      "the component helps**.\n")
    W("The older ablation table, taken from `full` (RevIN on), is a different "
      "configuration. It stays in `paper/tables/ablation.tex` as the appendix "
      "table on the contribution of each component with RevIN.\n")
    W("> **Row `− variable attention` removes two things.** With "
      "`use_var_attn=False` the variable attention is constant, so the entropy "
      "regulariser on it has no gradient path. The row is the joint removal of "
      "variable attention and the entropy term, not variable attention alone.\n")

    missing = []
    for ds, title in DATASETS:
        s = d[d.dataset == ds]
        b = s[s.ablation == BASE]
        W(f"## {title}\n")
        if b.empty:
            W("_headline runs missing_\n")
            missing.append(f"{ds}: {BASE}")
            continue
        true_f = os.path.join(PRED, f"{ds}_true.npy")
        true = np.load(true_f).astype(np.float64) if os.path.exists(true_f) else None
        ours = load(ds, BASE)

        # DM for every row first, so Holm can run over the family
        dm = {}
        for kind in ("abs", "sq"):
            raw = []
            for abl, _ in ROWS:
                ps = load(ds, abl)
                if true is None or not ours or not ps:
                    raw.append(np.nan)
                    continue
                md_, _, p, _ = sig.diebold_mariano(
                    sig.window_loss(ps, true, kind), sig.window_loss(ours, true, kind))
                dm[(abl, kind)] = md_
                raw.append(p)
            for (abl, _), pa in zip(ROWS, sig.holm(raw)):
                dm[(abl, kind, "p")] = pa

        W("| Variant | n | MAE | ΔMAE | RMSE | R² | DM ΔL1 (p_Holm) | DM ΔL2 (p_Holm) |")
        W("|---|---|---|---|---|---|---|---|")
        W(f"| **MeteoFormer (headline, no RevIN)** | {b.seed.nunique()} | "
          f"{b.MAE.mean():.3f} ± {b.MAE.std():.3f} | — | "
          f"{b.RMSE.mean():.3f} ± {b.RMSE.std():.3f} | "
          f"{b.R2.mean():.3f} ± {b.R2.std():.3f} | — | — |")
        for abl, label in ROWS:
            g = s[s.ablation == abl]
            if g.empty:
                missing.append(f"{ds}: {abl}")
                W(f"| {label} | 0 | _not run_ | | | | | |")
                continue
            cells = []
            for kind in ("abs", "sq"):
                if (abl, kind) in dm:
                    pa = dm[(abl, kind, "p")]
                    star = "\\*" if pa < 0.05 else ""
                    cells.append(f"{dm[(abl, kind)]:+.4f}{star} ({fp(pa)})")
                else:
                    cells.append("_predictions missing_")
            W(f"| {label} | {g.seed.nunique()} | "
              f"{g.MAE.mean():.3f} ± {g.MAE.std():.3f} | "
              f"{g.MAE.mean() - b.MAE.mean():+.3f} | "
              f"{g.RMSE.mean():.3f} ± {g.RMSE.std():.3f} | "
              f"{g.R2.mean():.3f} ± {g.R2.std():.3f} | " + " | ".join(cells) + " |")
        W("")
        W("DM: ablated minus headline per-window loss, averaged over seeds; "
          "**positive Δ means removing the component hurts**. `*` = p<0.05 after "
          "Holm over the six variants within this dataset and loss.\n")

    if missing:
        W("## Missing\n")
        for m in missing:
            W(f"- {m}")
        W("")
    dst = os.path.join(ROOT, "analysis", "ablation_norevin.md")
    open(dst, "w").write("\n".join(L))
    print("->", dst, "| missing:", missing or "none")


if __name__ == "__main__":
    main()
