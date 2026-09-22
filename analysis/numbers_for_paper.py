"""Collect every final number for the manuscript into one file.

Nothing here is retyped or summarised: LaTeX tables are copied verbatim out
of paper/tables/*.tex, markdown tables are copied verbatim out of the
analysis reports, and the one table the paper needs but no artefact holds in
full (all metrics at once, with parameter counts) is rebuilt from
analysis/results_clean.csv at 6 decimals.

Run after report.py, significance.py, per_target_horizon.py, frost_events.py
and the analysis reports, i.e. last.

Output -> analysis/NUMBERS_FOR_PAPER.md
"""

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from selection import selected_norm, historical_norm  # noqa: E402

TAB = os.path.join(ROOT, "paper", "tables")
AN = os.path.join(ROOT, "analysis")
OURS = "XAI-MeteoFormer"
LABEL = "MeteoFormer"
DATASETS = [("jena", "Jena"), ("beijing_aotizhongxin", "Beijing (Aotizhongxin)")]
METRICS = ["MAE", "RMSE", "MSE", "SMAPE", "R2"]

L = []
W = L.append


def tex(name, note=""):
    """copy a generated LaTeX table verbatim"""
    p = os.path.join(TAB, name)
    if not os.path.exists(p):
        W(f"_missing: `paper/tables/{name}` — regenerate it._\n")
        return
    body = open(p).read().strip()
    W(f"**`paper/tables/{name}`**{(' — ' + note) if note else ''}\n")
    W("```latex")
    W(body)
    W("```\n")


def md(name, start, end=None, note=""):
    """copy a section of a generated markdown report verbatim"""
    p = os.path.join(AN, name)
    if not os.path.exists(p):
        W(f"_missing: `analysis/{name}`._\n")
        return
    t = open(p).read()
    if start not in t:
        W(f"_missing section `{start}` in `analysis/{name}`._\n")
        return
    i = t.index(start)
    j = t.index(end, i + len(start)) if end and end in t[i + len(start):] else len(t)
    W(f"**`analysis/{name}`**{(' — ' + note) if note else ''}\n")
    W(t[i:j].strip() + "\n")


def full_main_table():
    """every metric at once, with parameter counts — not in any single .tex"""
    d = pd.read_csv(os.path.join(AN, "results_clean.csv"))
    d = d[(d.width == "default") & d.ablation.isin(["full", "no_revin"])]
    d = d[~((d.model == OURS) & (d.ablation == "full"))]
    keep = [m == OURS or selected_norm(m, ds) == nv
            for m, ds, nv in zip(d.model, d.dataset, d.norm_variant)]
    d = d[keep]
    for ds, title in DATASETS:
        x = d[d.dataset == ds]
        W(f"### {title} — every metric, mean ± sd over 5 seeds\n")
        W("| Model | Norm. | params | " + " | ".join(METRICS) + " | seeds |")
        W("|---|---|---|" + "---|" * (len(METRICS) + 1))
        g = x.groupby("model")
        order = g.MAE.mean().sort_values().index
        for m in order:
            r = g.get_group(m)
            nv = r.norm_variant.iloc[0]
            if m == OURS:
                mark = "off (ours, ablation no_revin)"
            elif nv != historical_norm(m):
                mark = f"**{nv}** — switched by the validation rule"
            else:
                mark = f"{nv} — validation rule keeps the historical variant"
            cells = [f"{r[k].mean():.6f} ± {r[k].std():.6f}" for k in METRICS]
            name = f"**{LABEL}**" if m == OURS else m
            W(f"| {name} | {mark} | {int(r.params.iloc[0]):,} | " +
              " | ".join(cells) + f" | {r.seed.nunique()} |")
        W("")
    W("Per-target and per-horizon columns of the same rows (`MAE_T`, `RMSE_T`, "
      "`MAPE_P`, `MAE_h1` …) are in `analysis/results_clean.csv`; the tables in "
      "section 3 below are generated from them.\n")


# --------------------------------------------------------------------------- #
W("# Every final number for the manuscript\n")
W("Assembled by [analysis/numbers_for_paper.py](analysis/numbers_for_paper.py) "
  "from the generated artefacts. LaTeX blocks are byte-for-byte copies of "
  "`paper/tables/*.tex`; markdown tables are copied verbatim from the analysis "
  "reports; the combined accuracy table in §1 is rebuilt from "
  "`analysis/results_clean.csv` at 6 decimals. Nothing is retyped, rounded "
  "further or paraphrased.\n")
W("Headline model: `XAI-MeteoFormer`, ablation `no_revin`, seq_len 96, "
  "HuberLoss(delta=1), 5 seeds. Baseline normalization is the "
  "validation-selected variant (`analysis/selection.py`); the historical "
  "variant is in `main_historical_*.tex`.\n")
W("---\n")

W("## 1. Main accuracy tables\n")
full_main_table()
tex("main_jena.tex", "as typeset, MAE/RMSE/R² only")
tex("main_beijing_aotizhongxin.tex", "as typeset")
tex("main_historical_jena.tex", "appendix: historical normalization")
tex("main_historical_beijing_aotizhongxin.tex", "appendix: historical normalization")

W("## 2. Significance\n")
tex("significance_jena.tex")
tex("significance_beijing_aotizhongxin.tex")
tex("significance_channels_jena.tex")
tex("significance_channels_beijing_aotizhongxin.tex")
W("Raw values, every baseline × loss × channel, including DM statistic, lag "
  "and unadjusted p: `analysis/significance_dm.csv`. Comparison against the "
  "historical normalization instead: `analysis/significance_historical*`.\n")

W("## 3. Per-target and per-horizon\n")
for met in ("mae", "rmse", "r2"):
    for ds, _ in DATASETS:
        tex(f"per_target_{met}_{ds}.tex")
for met in ("mae", "rmse", "r2"):
    for ds, _ in DATASETS:
        tex(f"per_horizon_{met}_{ds}.tex")

def mape_table():
    """MAPE exists only for P and RH (temperature crosses zero) and has no
    generated .tex; rebuilt from results_clean at 6 decimals."""
    d = pd.read_csv(os.path.join(AN, "results_clean.csv"))
    d = d[(d.width == "default") & d.ablation.isin(["full", "no_revin"])]
    d = d[~((d.model == OURS) & (d.ablation == "full"))]
    keep = [m == OURS or selected_norm(m, ds) == nv
            for m, ds, nv in zip(d.model, d.dataset, d.norm_variant)]
    d = d[keep]
    cols = [c for c in ("MAPE_RH", "MAPE_P") if c in d.columns]
    if not cols:
        return
    W("**MAPE per target** — reported only for relative humidity and pressure; "
      "temperature in Celsius crosses zero, so a percentage error is undefined "
      "there and sMAPE (§1) is reported instead.\n")
    for ds, title in DATASETS:
        x = d[d.dataset == ds]
        W(f"_{title}_\n")
        W("| Model | " + " | ".join(cols) + " |")
        W("|---|" + "---|" * len(cols))
        g = x.groupby("model")
        for m in g.MAE.mean().sort_values().index:
            r = g.get_group(m)
            name = f"**{LABEL}**" if m == OURS else m
            W(f"| {name} | " + " | ".join(
                f"{r[c].mean():.6f} ± {r[c].std():.6f}" for c in cols) + " |")
        W("")


mape_table()

W("## 4. Crossformer at full width, ablations, frost events\n")
md("crossformer_fullwidth.md", "## Accuracy by capacity", "## How to read this")
md("ablation_norevin.md", "## Jena", "## Missing")
tex("ablation_norevin.tex", "from the headline configuration")
tex("ablation.tex", "appendix: from `full`, Jena")
tex("events_jena.tex")
tex("events_beijing_aotizhongxin.tex")

W("## 5. Explanation fidelity (deterministic metric)\n")
md("xai_fidelity_v2.md", "## Jena", "## P0-4")
md("xai_fidelity_v2.md", "## P0-4", "## What does move faithfulness")
md("xai_fidelity_v2.md", "## What does move faithfulness", "\n## ")
tex("fidelity_jena.tex")
tex("fidelity_beijing_aotizhongxin.tex")
tex("fidelity_perm_appendix_jena.tex", "appendix: permutation metric")
tex("fidelity_perm_appendix_beijing_aotizhongxin.tex", "appendix: permutation metric")

W("## 6. Robustness, transfer, window length, calibration\n")
md("occlusion_time.md", "## Headline model", "## Verdict",
   "all four configurations: headline, the clean entropy control, full, no_entropy")
md("block_missing.md", "## Jena", "## Verdict")
md("missing_robustness.md", "## Jena", "## Verdict")
md("aug_robustness.md", "## Jena", "## Reading")
md("cross_station.md", "## Mean over the 11 target stations", "## MAE per station")
md("cross_station.md", "## MAE per station", "\n## ")
md("window_sweep.md", "## Jena", "## Reading")
md("variance_calibration.md", "## Jena", "## Reading")
md("loss_mse.md", "## Jena", "## Reading", "appendix: MSE instead of Huber")

W("## 7. Attention stability (for the reconstructed Table 6)\n")
md("attention_stability.md", "## Jena", "## Verdict")

dst = os.path.join(AN, "NUMBERS_FOR_PAPER.md")
open(dst, "w").write("\n".join(L))
print("->", dst, f"({len(L)} blocks)")
