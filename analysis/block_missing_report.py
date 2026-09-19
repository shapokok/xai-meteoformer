"""Render analysis/block_missing.md from analysis/block_missing.csv."""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OURS, LABEL = "XAI-MeteoFormer", "MeteoFormer"
NAMES = ["T", "RH", "P", "WS"]


def nm(m):
    return f"**{LABEL}**" if m == OURS else m


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selection import keep_selected  # noqa: E402


def main():
    d = pd.read_csv(os.path.join(ROOT, "analysis", "block_missing.csv"))
    d = keep_selected(d)     # the variant the main table reports
    L = []
    W = L.append
    W("# Block gaps: an operational outage, and a second external check (P0-5)\n")
    W("Produced by [analysis/block_missing.py](analysis/block_missing.py) and "
      "[analysis/block_missing_report.py](analysis/block_missing_report.py). "
      "Inference only on the existing checkpoints; 11 models × 5 seeds × both "
      "datasets, on the same test subsample as the per-cell experiment "
      f"({d.groupby('dataset').n_windows.first().to_dict()} windows).\n")
    W("## Method\n")
    W("A contiguous block of **16 hours is removed from every channel** of the "
      "96-hour input window — a station outage, not scattered sensor dropouts. "
      "Two placements with an **identical missing fraction** (16/96 = 16.7% of "
      "input cells):\n")
    W("- **A — end of the window** (t = 80…96, the most recent 16 h)")
    W("- **B — start of the window** (t = 0…16, the oldest 16 h)\n")
    W("Repair uses the same rule as the per-cell experiment: linear "
      "interpolation along time over the surviving indices, per channel. For "
      "a block at the end of the window there is nothing to interpolate "
      "towards, so this degenerates to **holding the last valid value** "
      "(t = 79) for 16 hours — exactly what an operational forward-fill does "
      "after an outage. For a block at the start it holds the first valid "
      "value backwards.\n")
    W("This is also a second external test of "
      "[analysis/occlusion_time.md](analysis/occlusion_time.md), which found "
      "that our model uses essentially only the last 16 hours. If so, A must "
      "hurt far more than B.\n")

    for ds, title in (("jena", "Jena"), ("beijing_aotizhongxin",
                                         "Beijing (Aotizhongxin)")):
        x = d[d.dataset == ds]
        W(f"\n---\n\n## {title}\n")
        W("### MAE, mean ± sd over 5 seeds\n")
        W("| Model | intact | A: last 16 h lost | B: first 16 h lost | "
          "Δ A | Δ B |")
        W("|---|---|---|---|---|---|")
        rows = []
        for m, g in x.groupby("model"):
            v = {k: g[g.variant == k].sort_values("seed").MAE.values
                 for k in ("none", "A_end", "B_start")}
            dA = 100 * (v["A_end"] - v["none"]) / v["none"]
            dB = 100 * (v["B_start"] - v["none"]) / v["none"]
            rows.append((dA.mean(), m, v, dA, dB))
        for _, m, v, dA, dB in sorted(rows):
            W(f"| {nm(m)} | {v['none'].mean():.3f}±{v['none'].std(ddof=1):.3f} | "
              f"{v['A_end'].mean():.3f}±{v['A_end'].std(ddof=1):.3f} | "
              f"{v['B_start'].mean():.3f}±{v['B_start'].std(ddof=1):.3f} | "
              f"**{dA.mean():+.1f}%** | {dB.mean():+.1f}% |")
        W("")

        # A vs B, paired over (model, seed)
        a = np.concatenate([r[3] for r in rows])
        b = np.concatenate([r[4] for r in rows])
        t = stats.wilcoxon(a, b)
        W(f"Across all {len(a)} (model, seed) pairs, losing the **last** 16 h "
          f"costs **{a.mean():+.1f}%** MAE on average, losing the **first** "
          f"16 h costs **{b.mean():+.1f}%**. Paired Wilcoxon A vs B: "
          f"p = {t.pvalue:.1e}. A is worse than B for "
          f"**{int((a > b).sum())} of {len(a)}** pairs.\n")

        rk0 = x[x.variant == "none"].groupby("model").MAE.mean().sort_values()
        rkA = x[x.variant == "A_end"].groupby("model").MAE.mean().sort_values()
        W("### Does first place survive an outage in the last 16 h?\n")
        W("| Rank | intact | A: last 16 h lost |")
        W("|---|---|---|")
        for i, (p, q) in enumerate(zip(rk0.index, rkA.index), 1):
            W(f"| {i} | {nm(p)} {rk0[p]:.3f} | {nm(q)} {rkA[q]:.3f} |")
        pos = list(rkA.index).index(OURS) + 1
        W(f"\nUnder an outage in the last 16 h our model falls from **1st to "
          f"{pos}th** by MAE.\n")

        W("### Per target, Δ% of MAE under A (last 16 h lost)\n")
        W("| Model | " + " | ".join(NAMES) + " |")
        W("|---|" + "---|" * 4)
        for m, g in sorted(x.groupby("model")):
            cs = []
            for n in NAMES:
                b0 = g[g.variant == "none"][f"MAE_{n}"].mean()
                bA = g[g.variant == "A_end"][f"MAE_{n}"].mean()
                cs.append(f"{100*(bA-b0)/b0:+.0f}")
            W(f"| {nm(m)} | " + " | ".join(cs) + " |")
        W("")

    W("\n---\n\n## Verdict\n")
    W("**1. The occlusion finding is confirmed — and it is not specific to our "
      "model.** Every one of the 11 models is hurt far more by losing the most "
      "recent 16 hours than by losing the oldest 16 hours of the same window, "
      "at the same missing fraction. Relying on the recent past is a property "
      "of hourly weather forecasting at a 24 h horizon (persistence is a strong "
      "predictor), not an idiosyncrasy of our architecture. That makes "
      "occlusion_time.md *less* damaging than it first looked: \"the model "
      "uses the last 16 hours\" is what every model here does.\n")
    W("**2. Our model is the most recency-concentrated of all.** It is among the "
      "most damaged by losing the recent block and the least damaged by losing "
      "the old one. Jena: +73% vs +0.6%; Beijing: +60% vs +0.4%. It extracts "
      "almost nothing from hours 0–80 of its window, more so than any "
      "baseline.\n")
    W("**3. First place does not survive an outage in the last 16 hours.** On "
      "both datasets our model drops out of the top by MAE once the most recent "
      "block is lost and forward-filled. This is the operationally relevant "
      "failure mode — a station outage — and it must be reported, not only the "
      "per-cell result where every model looked robust.\n")
    W("**4. What it means for the paper.** The per-cell experiment "
      "([missing_robustness.md](analysis/missing_robustness.md)) and this one "
      "answer different questions and should be reported together: scattered "
      "dropouts are harmless for everyone because interpolation repairs them; a "
      "recent outage is severe for everyone and worst for us. Combined with the "
      "window-length sweep (P0-2, GPU track), this points to a concrete, "
      "testable design consequence — our model's 96-hour context is largely "
      "unused, and its accuracy advantage is bought with maximal dependence on "
      "the freshest observations.\n")
    W("## Caveat on devices\n")
    W("For PatchTST and our model, part of the Jena rows were computed on CPU "
      "before the MPS `unfold` workaround and the rest on MPS after it. The two "
      "paths agree to ~2·10⁻⁶ on outputs and ~6·10⁻⁷ relative on gradients "
      "(verified on real checkpoints, see "
      "[analysis/xai_patches.py](analysis/xai_patches.py)); the mix has no "
      "material effect on any number here.\n")

    dst = os.path.join(ROOT, "analysis", "block_missing.md")
    open(dst, "w").write("\n".join(L))
    print("->", dst)


if __name__ == "__main__":
    main()
