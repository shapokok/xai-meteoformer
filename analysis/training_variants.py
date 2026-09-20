"""Two training-recipe variants, both swept over ALL 11 models.

The published recipe is HuberLoss(delta=1) without augmentation. Two
variants were trained on the GPU track, each for every model, 5 seeds, both
datasets, so neither is a private advantage for the proposed model:

  loss=mse      the criterion RMSE reports (analysis/loss_mse.md)
  aug_block=0.5 station-outage augmentation (analysis/aug_robustness.md)

Neither replaces the headline: the selection rule fixed in
analysis/norm_symmetry_runs.md is the lower mean VALIDATION loss, and on
clean validation data both variants are worse for our model. They are
reported as a robustness mitigation and as a negative result.

Reads analysis/results_variants.csv (written by repair_results_csv.py),
analysis/results_clean.csv, analysis/block_missing.csv and predictions/.

Outputs -> analysis/aug_robustness.md, analysis/loss_mse.md
        -> paper/tables/aug_robustness.tex, paper/tables/loss_mse_appendix.tex
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from selection import selected_ablation, selected_norm, selected_suffix  # noqa: E402

PRED = os.path.join(ROOT, "predictions")
OURS = "XAI-MeteoFormer"
LABEL = "MeteoFormer"
DATASETS = [("jena", "Jena"), ("beijing_aotizhongxin", "Beijing (Aotizhongxin)")]
NAMES = ["T", "RH", "P", "WS"]


def published():
    d = pd.read_csv(os.path.join(ROOT, "analysis", "results_clean.csv"))
    d = d[(d.width == "default") & d.ablation.isin(["full", "no_revin"])]
    d = d[~((d.model == OURS) & (d.ablation == "full"))]
    # results_clean keeps the normalization in its own column, so the filter
    # is on norm_variant -- not on the tag suffix the harness CSVs carry.
    keep = [m == OURS or selected_norm(m, ds) == nv for m, ds, nv in
            zip(d.model, d.dataset, d.norm_variant)]
    return d[keep]


def variants():
    d = pd.read_csv(os.path.join(ROOT, "analysis", "results_variants.csv"))
    d = d[d.ablation.isin(["full", "no_revin"]) & (d.width == "default")]
    keep = [m == OURS or selected_norm(m, ds) == nv for m, ds, nv in
            zip(d.model, d.dataset, d.norm_variant)]
    return d[keep]


def name(m):
    return f"**{LABEL}**" if m == OURS else m


def table(W, base, var, metric, extra=""):
    """one row per model: published recipe vs the variant, with both ranks"""
    t = base.groupby("model")[metric].mean().to_frame("base").join(
        var.groupby("model")[metric].mean().rename("var"))
    t["rank_base"] = t["base"].rank()
    t["rank_var"] = t["var"].rank()
    t = t.sort_values("var")
    W(f"| Model | {metric} published | {metric} variant | Δ | rank published | rank variant |")
    W("|---|---|---|---|---|---|")
    for m, r in t.iterrows():
        if pd.isna(r["var"]):
            W(f"| {name(m)} | {r['base']:.3f} | _missing_ | | {int(r.rank_base)} | |")
            continue
        W(f"| {name(m)} | {r['base']:.3f} | {r['var']:.3f} | {r['var'] - r['base']:+.3f} | "
          f"{int(r.rank_base)} | {int(r.rank_var)} |")
    W("")
    return t


def sd_ratio(ds, model, abl, suffix, ch):
    true = np.load(os.path.join(PRED, f"{ds}_true.npy")).astype(np.float64)
    out = []
    for s in range(5):
        f = os.path.join(PRED, f"{model}_{ds}_{abl}{suffix}_s{s}_pred.npy")
        if os.path.exists(f):
            p = np.load(f).astype(np.float64)
            out.append(p[:, :, ch].std() / true[:, :, ch].std())
    return float(np.mean(out)) if out else float("nan")


def oracle_a(ds, model, abl, suffix, ch):
    """scalar a minimising MSE under pred -> mean + a*(pred-mean), on test;
    a measurement of over-dispersion, not a method"""
    true = np.load(os.path.join(PRED, f"{ds}_true.npy")).astype(np.float64)[:, :, ch]
    out = []
    for s in range(5):
        f = os.path.join(PRED, f"{model}_{ds}_{abl}{suffix}_s{s}_pred.npy")
        if not os.path.exists(f):
            continue
        p = np.load(f).astype(np.float64)[:, :, ch]
        c = p - p.mean()
        out.append(float((c * (true - p.mean())).sum() / (c * c).sum()))
    return float(np.mean(out)) if out else float("nan")


# --------------------------------------------------------------------------- #
def aug_report(base, var):
    blk = pd.read_csv(os.path.join(ROOT, "analysis", "block_missing.csv"))
    blk["fam"] = np.where(blk.ablation.str.contains("augblk"), "aug", "base")
    blk["stem"] = blk.ablation.str.replace("_augblk0.5", "", regex=False)
    blk = blk[[selected_ablation(m, ds) == a for m, ds, a in
               zip(blk.model, blk.dataset, blk.stem)]]

    L = []
    W = L.append
    W("# Station-outage augmentation (robustness)\n")
    W("Produced by [analysis/training_variants.py](analysis/training_variants.py). "
      "During training, with probability 0.5 one contiguous 16 h block of the "
      "input window is blanked across every channel and refilled by linear "
      "interpolation — the repair an operational forward-fill performs — at a "
      "uniformly random position (`block_dropout` in [src/train.py](src/train.py)). "
      "**Every one of the 11 models is retrained with it**, 5 seeds, both "
      "datasets, so the comparison stays symmetric.\n")
    W("The outage test is the one in [analysis/block_missing.md](analysis/block_missing.md): "
      "variant A removes the most recent 16 h of the input window at inference.\n")
    W("**This variant is not the headline model.** The selection rule is the "
      "lower mean validation loss, and on clean validation data the augmented "
      "model is slightly worse (see the table below). Choosing it because it "
      "wins the outage test would be selection on test.\n")

    for ds, title in DATASETS:
        b, v = base[base.dataset == ds], var[(var.dataset == ds) & (var.aug_block > 0)]
        ours_b = b[b.model == OURS]
        ours_v = v[v.model == OURS]
        x = blk[blk.dataset == ds]
        W(f"## {title}\n")
        W("### Our model: clean vs outage\n")
        W("| Recipe | validation loss | clean MAE | outage MAE | degradation | rank under outage |")
        W("|---|---|---|---|---|---|")
        for fam, rows, label in (("base", ours_b, "published (Huber, no augmentation)"),
                                 ("aug", ours_v, "with outage augmentation")):
            g = x[x.fam == fam]
            if g.empty or rows.empty:
                W(f"| {label} | _missing_ | | | | |")
                continue
            per = g.groupby(["model", "variant"]).MAE.mean()
            clean = per[(OURS, "none")]
            out = per[(OURS, "A_end")]
            rk = int(g[g.variant == "A_end"].groupby("model").MAE.mean().rank()[OURS])
            n = g[g.variant == "A_end"].model.nunique()
            W(f"| {label} | {rows.best_val.mean():.4f} | {clean:.3f} | {out:.3f} | "
              f"+{100 * (out / clean - 1):.0f}% | {rk} / {n} |")
        W("")
        W("### Every model under the outage (MAE, variant A: last 16 h lost)\n")
        pv = x[x.variant == "A_end"].groupby(["model", "fam"]).MAE.mean().unstack()
        if {"base", "aug"} <= set(pv.columns):
            pv["rank_base"] = pv["base"].rank()
            pv["rank_aug"] = pv["aug"].rank()
            pv = pv.sort_values("aug")
            W("| Model | published | augmented | Δ | rank published | rank augmented |")
            W("|---|---|---|---|---|---|")
            for m, r in pv.iterrows():
                if pd.isna(r.get("aug", np.nan)):
                    W(f"| {name(m)} | {r['base']:.3f} | _missing_ | | {int(r.rank_base)} | |")
                    continue
                W(f"| {name(m)} | {r['base']:.3f} | {r['aug']:.3f} | {r['aug'] - r['base']:+.3f} | "
                  f"{int(r.rank_base)} | {int(r.rank_aug)} |")
            W("")
        W("### Accuracy on clean data, every model\n")
        table(W, b, v, "MAE")

    W("## Reading\n")
    W("- On Jena the augmentation removes the weakness outright: our model goes "
      "from 6th to 1st under the outage, and clean accuracy is unchanged.\n")
    W("- On Beijing it helps but does not change the standing: the degradation "
      "falls from +60% to +48%, every other model improves too, our rank under "
      "the outage stays 8th, and clean MAE gets worse (4.151 → 4.255) while "
      "keeping 1st place.\n")
    W("- Both datasets are reported. Showing Jena alone would be selective "
      "reporting of the same experiment.\n")
    open(os.path.join(ROOT, "analysis", "aug_robustness.md"), "w").write("\n".join(L))

    # paper table: our model only, both datasets, clean vs outage
    lines = ["\\begin{tabular}{llcccc}", "\\toprule",
             "Dataset & Recipe & Val.\\ loss & Clean MAE & Outage MAE & Rank under outage \\\\",
             "\\midrule"]
    for ds, title in DATASETS:
        x = blk[blk.dataset == ds]
        nm = title.split(" ")[0]
        for fam, rows, label in (
                ("base", base[(base.dataset == ds) & (base.model == OURS)], "published"),
                ("aug", var[(var.dataset == ds) & (var.model == OURS) & (var.aug_block > 0)],
                 "+ outage augmentation")):
            g = x[x.fam == fam]
            if g.empty or rows.empty:
                continue
            per = g.groupby(["model", "variant"]).MAE.mean()
            rk = int(g[g.variant == "A_end"].groupby("model").MAE.mean().rank()[OURS])
            n = g[g.variant == "A_end"].model.nunique()
            lines.append(f"{nm} & {label} & {rows.best_val.mean():.4f} & "
                         f"{per[(OURS, 'none')]:.3f} & {per[(OURS, 'A_end')]:.3f} & {rk}/{n} \\\\")
            nm = ""
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    with open(os.path.join(ROOT, "paper", "tables", "aug_robustness.tex"), "w") as f:
        f.write("\\begin{table}[H]\n\\caption{Station-outage augmentation, applied "
                "to all 11 models during training (p=0.5, one 16 h block per "
                "window). Outage MAE: the most recent 16 h of the input window "
                "are lost at inference. The augmented model is not the headline: "
                "it is worse on clean validation loss, which is the selection "
                "rule.}\n\\label{tab:aug_robustness}\n" + "\n".join(lines) +
                "\n\\end{table}\n")


def mse_report(base, var):
    L = []
    W = L.append
    W("# Training with MSE instead of Huber (appendix, negative result)\n")
    W("Produced by [analysis/training_variants.py](analysis/training_variants.py). "
      "Every published run uses `HuberLoss(delta=1)`. Huber linearises the "
      "largest residuals, and RMSE lives exactly in that tail, so training with "
      "MSE is the direct way to test whether our RMSE gap to Crossformer is a "
      "loss-function artefact. **All 11 models** were retrained with MSE, 5 "
      "seeds, both datasets.\n")
    W("**Result: it does not close the gap, and on Beijing it costs us first "
      "place on MAE.** The variant is not adopted.\n")
    for ds, title in DATASETS:
        b, v = base[base.dataset == ds], var[(var.dataset == ds) & (var.loss == "mse")]
        W(f"## {title}\n")
        W("### MAE\n")
        table(W, b, v, "MAE")
        W("### RMSE\n")
        table(W, b, v, "RMSE")
        W("### Why it falls short: dispersion of the humidity forecast\n")
        W("`sd ratio RH` = sd of the predicted RH over sd of the observed RH. "
          "A squared loss is minimised below 1 (the oracle column, fitted on "
          "test, is a measurement of the over-dispersion, not a method).\n")
        ch = NAMES.index("RH")
        rows = [("published (Huber)", OURS, "no_revin", ""),
                ("with MSE", OURS, "no_revin", "_mse"),
                ("Crossformer (published)", "Crossformer", "full",
                 selected_suffix("Crossformer", ds))]
        W("| Variant | sd ratio RH | oracle a (RH) |")
        W("|---|---|---|")
        for label, m, abl, sfx in rows:
            W(f"| {label} | {sd_ratio(ds, m, abl, sfx, ch):.3f} | "
              f"{oracle_a(ds, m, abl, sfx, ch):.3f} |")
        W("")
    W("## Reading\n")
    W("The mechanism is real but the effect is too small: MSE moves our RH "
      "dispersion only part of the way towards the optimum, while Crossformer "
      "is already close to it and improves as well. Per-channel budgets are in "
      "`analysis/_tables_mse_{dataset}.md` "
      "([analysis/error_decomposition.py](analysis/error_decomposition.py)).\n")
    open(os.path.join(ROOT, "analysis", "loss_mse.md"), "w").write("\n".join(L))

    lines = ["\\begin{tabular}{llcccc}", "\\toprule",
             "Dataset & Model & MAE (Huber) & MAE (MSE) & RMSE (Huber) & RMSE (MSE) \\\\",
             "\\midrule"]
    for ds, title in DATASETS:
        nm = title.split(" ")[0]
        b, v = base[base.dataset == ds], var[(var.dataset == ds) & (var.loss == "mse")]
        bm = b.groupby("model")[["MAE", "RMSE"]].mean()
        vm = v.groupby("model")[["MAE", "RMSE"]].mean()
        for m in [OURS] + [x for x in bm.index if x != OURS]:
            if m not in vm.index:
                continue
            label = "\\textbf{" + LABEL + "}" if m == OURS else m
            lines.append(f"{nm} & {label} & {bm.MAE[m]:.3f} & {vm.MAE[m]:.3f} & "
                         f"{bm.RMSE[m]:.3f} & {vm.RMSE[m]:.3f} \\\\")
            nm = ""
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    with open(os.path.join(ROOT, "paper", "tables", "loss_mse_appendix.tex"), "w") as f:
        f.write("\\begin{table}[H]\n\\caption{Appendix: every model retrained with "
                "MSE instead of the Huber loss used throughout the paper, 5 seeds. "
                "MSE does not close our RMSE gap to Crossformer and costs first "
                "place on MAE on Beijing, so the published recipe is kept.}\n"
                "\\label{tab:loss_mse}\n" + "\n".join(lines) + "\n\\end{table}\n")


def main():
    base, var = published(), variants()
    aug_report(base, var)
    mse_report(base, var)
    print("-> analysis/aug_robustness.md, analysis/loss_mse.md")
    print("-> paper/tables/aug_robustness.tex, paper/tables/loss_mse_appendix.tex")


if __name__ == "__main__":
    main()
