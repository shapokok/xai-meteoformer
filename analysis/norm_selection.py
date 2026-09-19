"""Normalization symmetry for the baselines (P1-1): results and selection.

Applies the rule written down in advance in analysis/norm_symmetry_runs.md:
per (model, dataset) pick the normalization variant with the lower mean
VALIDATION loss over the 5 seeds, the same rule that selected no_revin for
our model. Test is only read.

Reports, per dataset:
  - both variants side by side (val loss, test MAE/RMSE/R2),
  - which variant the rule selects and whether that differs from the
    historical configuration used in the published tables,
  - Diebold-Mariano of the headline model against the SELECTED variant,
    reusing analysis/significance.py, Holm within the family.

The authors adopted the rule for the main table: paper/tables/main_*.tex
(src/report.py) and every downstream analysis use the selected variants
through analysis/selection.py; the historical configuration is in
paper/tables/main_historical_*.tex.

Outputs -> analysis/norm_selection.md
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
SWEPT = ["DLinear", "Transformer", "Informer", "Autoformer", "LSTM"]
HIST = {"LSTM": "on"}                       # every other baseline: "off"
OURS = ("XAI-MeteoFormer", "no_revin")
DATASETS = [("jena", "Jena"), ("beijing_aotizhongxin", "Beijing (Aotizhongxin)")]


def tag_suffix(model, nv):
    return "" if nv == HIST.get(model, "off") else f"_norm{nv}"


def load(model, abl, ds, suffix=""):
    ps = []
    for s in range(5):
        f = os.path.join(PRED, f"{model}_{ds}_{abl}{suffix}_s{s}_pred.npy")
        if os.path.exists(f):
            ps.append(np.load(f).astype(np.float64))
    return ps


def main():
    d = pd.read_csv(os.path.join(ROOT, "analysis", "results_clean.csv"))
    d = d[(d.width == "default")]
    L = []
    W = L.append
    W("# Normalization symmetry for the baselines (P1-1)\n")
    W("Produced by [analysis/norm_selection.py](analysis/norm_selection.py). "
      "Selection rule, fixed in advance in "
      "[analysis/norm_symmetry_runs.md](analysis/norm_symmetry_runs.md): per "
      "(model, dataset) the variant with the lower mean **validation** loss over "
      "5 seeds. `on` = the same RevIN our model can use, added to the baseline; "
      "`off` = the baseline as its source defines it (LSTM: without RevIN). "
      "Historical = what the published tables use (LSTM `on`, others `off`).\n")
    missing, all_sel = [], {}
    for ds, title in DATASETS:
        W(f"## {title}\n")
        W("| Model | variant | n | val loss | MAE | RMSE | R² | selected |")
        W("|---|---|---|---|---|---|---|---|")
        sel = {}
        for m in SWEPT:
            g = d[(d.model == m) & (d.dataset == ds) & (d.ablation == "full")]
            vals = {}
            for nv in ("off", "on"):
                x = g[g.norm_variant == nv]
                if x.empty:
                    missing.append(f"{ds} {m} norm={nv}")
                    continue
                vals[nv] = x
            if len(vals) < 2:
                sel[m] = HIST.get(m, "off")
            else:
                sel[m] = min(vals, key=lambda k: vals[k].best_val.mean())
            for nv, x in vals.items():
                mark = []
                if nv == sel[m]:
                    mark.append("**selected**")
                if nv == HIST.get(m, "off"):
                    mark.append("historical")
                W(f"| {m} | {nv} | {x.seed.nunique()} | {x.best_val.mean():.4f} | "
                  f"{x.MAE.mean():.3f} ± {x.MAE.std():.3f} | "
                  f"{x.RMSE.mean():.3f} ± {x.RMSE.std():.3f} | "
                  f"{x.R2.mean():.3f} ± {x.R2.std():.3f} | {', '.join(mark)} |")
        all_sel[ds] = sel
        changed = [m for m in SWEPT if sel[m] != HIST.get(m, "off")]
        W("")
        W("Selection differs from the historical configuration for: " +
          (", ".join(f"**{m}** (→ `{sel[m]}`)" for m in changed) if changed else "none") + ".\n")

        # accuracy table with the selected variants, and DM against them
        ours_rows = d[(d.model == OURS[0]) & (d.ablation == OURS[1]) & (d.dataset == ds)]
        true_f = os.path.join(PRED, f"{ds}_true.npy")
        true = np.load(true_f).astype(np.float64) if os.path.exists(true_f) else None
        ours = load(*OURS, ds)
        W("### Headline model vs the selected variants\n")
        W("| Baseline (selected) | MAE | Δ MAE (ours − baseline) | DM ΔL1 (p_Holm) | DM ΔL2 (p_Holm) |")
        W("|---|---|---|---|---|")
        res = {}
        for kind in ("abs", "sq"):
            raw = []
            for m in SWEPT:
                bp = load(m, "full", ds, tag_suffix(m, sel[m]))
                if true is None or not ours or not bp:
                    raw.append(np.nan)
                    if not bp:
                        missing.append(f"{ds} {m} predictions ({sel[m]})")
                    continue
                md_, _, p, _ = sig.diebold_mariano(sig.window_loss(ours, true, kind),
                                                   sig.window_loss(bp, true, kind))
                res[(m, kind)] = md_
                raw.append(p)
            for m, pa in zip(SWEPT, sig.holm(raw)):
                res[(m, kind, "p")] = pa
        for m in SWEPT:
            x = d[(d.model == m) & (d.dataset == ds) & (d.ablation == "full") &
                  (d.norm_variant == sel[m])]
            cells = []
            for kind in ("abs", "sq"):
                if (m, kind) in res:
                    pa = res[(m, kind, "p")]
                    star = "\\*" if pa < 0.05 else ""
                    cells.append(f"{res[(m, kind)]:+.4f}{star} ({fp(pa)})")
                else:
                    cells.append("_missing_")
            W(f"| {m} (`{sel[m]}`) | {x.MAE.mean():.3f} | "
              f"{ours_rows.MAE.mean() - x.MAE.mean():+.3f} | " + " | ".join(cells) + " |")
        W("")
        W(f"Headline MAE: {ours_rows.MAE.mean():.3f} ± {ours_rows.MAE.std():.3f}. "
          "Negative Δ favours the headline model. `*` = p<0.05 after Holm over "
          "the five baselines within each loss.\n")


    if missing:
        W("## Missing\n")
        for m in sorted(set(missing)):
            W(f"- {m}")
        W("")
    dst = os.path.join(ROOT, "analysis", "norm_selection.md")
    open(dst, "w").write("\n".join(L))
    print("->", dst, "| selected:", all_sel, "| missing:", sorted(set(missing)) or "none")


if __name__ == "__main__":
    main()
