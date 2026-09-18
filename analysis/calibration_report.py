"""Build analysis/variance_calibration.md from the calibrated predictions."""

import os

import numpy as np
import pandas as pd
from scipy import stats

import variance_calibration as vc

ROOT = vc.ROOT
NAMES = vc.NAMES
OURS = ("XAI-MeteoFormer", "no_revin")
RIVAL = ("Crossformer", "full")


def ms(v):
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.3f}±{v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"


def main():
    df, percase = vc.main()
    L = []
    W = L.append

    W("# Variance calibration fitted on validation, applied to test\n")
    W("Produced by [analysis/variance_calibration.py](analysis/variance_calibration.py) "
      "and [analysis/calibration_report.py](analysis/calibration_report.py), from "
      "validation predictions dumped by "
      "[analysis/dump_val_preds.py](analysis/dump_val_preds.py).\n")
    W("For every (model, ablation, seed, channel) one scalar `a` is fitted in "
      "closed form on the **validation** split, minimising validation MSE of "
      "`m + a*(pred - m)` where `m` is the mean of the validation predictions. "
      "`a` and `m` are then frozen and applied to the test predictions, which "
      "are scored once. No quantity is fitted, selected or thresholded on test. "
      "`a < 1` = shrink toward the mean, which is what a squared loss rewards.\n")
    W("Mean±sd over 5 seeds, physical units.\n")

    for ds in vc.DATASETS:
        d = df[df.dataset == ds]
        if not len(d):
            continue
        title = "Jena" if ds == "jena" else "Beijing (Aotizhongxin)"
        W(f"\n---\n\n## {title}\n")

        # ---- fitted coefficients ----
        W("### Fitted coefficients `a` (from validation)\n")
        W("| Model | " + " | ".join(NAMES) + " |")
        W("|---|" + "---|" * len(NAMES))
        for (m, a_), g in sorted(d.groupby(["model", "ablation"])):
            lab = m if a_ == "full" else f"{m} ({a_})"
            W(f"| {lab} | " + " | ".join(ms(g[f"a_{n}"]) for n in NAMES) + " |")
        W("")

        # ---- aggregate before/after ----
        W("### Aggregate test metrics, before → after calibration\n")
        W("| Model | MAE before | MAE after | RMSE before | RMSE after | R² before | R² after |")
        W("|---|---|---|---|---|---|---|")
        agg = []
        for (m, a_), g in sorted(d.groupby(["model", "ablation"])):
            lab = m if a_ == "full" else f"{m} ({a_})"
            agg.append((g.post_RMSE.mean(), lab, g))
            W(f"| {lab} | {ms(g.pre_MAE)} | {ms(g.post_MAE)} | "
              f"{ms(g.pre_RMSE)} | {ms(g.post_RMSE)} | "
              f"{ms(g.pre_R2)} | {ms(g.post_R2)} |")
        W("")
        W("Ranked by RMSE after calibration:\n")
        W("| # | Model | MAE | RMSE | R² |")
        W("|---|---|---|---|---|")
        for i, (_, lab, g) in enumerate(sorted(agg), 1):
            W(f"| {i} | {lab} | {ms(g.post_MAE)} | {ms(g.post_RMSE)} | {ms(g.post_R2)} |")
        W("")

        # ---- per target ----
        for met in ("MAE", "RMSE", "R2"):
            W(f"### Per-target {met}, before → after\n")
            W("| Model | " + " | ".join(f"{n} before → after" for n in NAMES) + " |")
            W("|---|" + "---|" * len(NAMES))
            for (m, a_), g in sorted(d.groupby(["model", "ablation"])):
                lab = m if a_ == "full" else f"{m} ({a_})"
                cells = [f"{ms(g[f'pre_{met}_{n}'])} → {ms(g[f'post_{met}_{n}'])}"
                         for n in NAMES]
                W(f"| {lab} | " + " | ".join(cells) + " |")
            W("")

        # ---- Wilcoxon, ours vs Crossformer, after calibration ----
        W("### Wilcoxon signed-rank: ours vs Crossformer, after calibration\n")
        ours = d[(d.model == OURS[0]) & (d.ablation == OURS[1])].sort_values("seed")
        riv = d[(d.model == RIVAL[0]) & (d.ablation == RIVAL[1])].sort_values("seed")
        if not len(ours) or not len(riv):
            W("_variant missing on this dataset_\n")
            continue

        W("**(a) Paired over seeds (n=5)** — the direct analogue of the paper's "
          "per-seed test. Note the floor: with n=5 the smallest attainable "
          "two-sided p is 0.0625, so this test *cannot* reach p<0.05 no matter "
          "how large the effect. Reported for completeness, not as evidence.\n")
        W("| Metric | ours | Crossformer | Δ (ours − rival) | W | p |")
        W("|---|---|---|---|---|---|")
        for met in ("MAE", "RMSE", "R2"):
            x = ours[f"post_{met}"].values
            y = riv[f"post_{met}"].values
            n = min(len(x), len(y))
            try:
                w, p = stats.wilcoxon(x[:n], y[:n])
                wp = f"{w:.1f} | {p:.4f}"
            except ValueError as e:
                wp = f"— | n/a ({e})"
            W(f"| {met} | {ms(x)} | {ms(y)} | {np.mean(x[:n]-y[:n]):+.4f} | {wp} |")
        W("")

        W("**(b) Paired over test windows** — per-window absolute/squared error, "
          "averaged over the 5 seeds of each model, paired window by window. "
          "This is the powerful version and the one worth quoting.\n")
        W("| Metric | mean ours | mean Crossformer | Δ | W | p | median Δ |")
        W("|---|---|---|---|---|---|---|")

        def stack(model, abl, idx):
            arr = [percase[(ds, model, abl, s)][idx] for s in range(5)
                   if (ds, model, abl, s) in percase]
            return np.mean(arr, axis=0), percase[(ds, model, abl, 0)][2]

        po, ttrue = stack(*OURS, 1)
        pr, _ = stack(*RIVAL, 1)
        for met, fn in (("MAE", lambda e: np.abs(e).mean(axis=(1, 2))),
                        ("MSE", lambda e: (e ** 2).mean(axis=(1, 2)))):
            eo, er = fn(po - ttrue), fn(pr - ttrue)
            w, p = stats.wilcoxon(eo, er)
            W(f"| per-window {met} | {eo.mean():.4f} | {er.mean():.4f} | "
              f"{eo.mean()-er.mean():+.4f} | {w:.0f} | {p:.3g} | "
              f"{np.median(eo-er):+.4f} |")
        W("")
        W(f"n = {len(ttrue)} test windows. Negative Δ favours our model.\n")

    dst = os.path.join(ROOT, "analysis", "variance_calibration.md")
    open(dst, "w").write("\n".join(L))
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
