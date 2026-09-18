"""Render analysis/missing_robustness.md."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ["T", "RH", "P", "WS"]
RATES = [0.0, 0.05, 0.10, 0.20]
LABEL, OURS = "MeteoFormer", "XAI-MeteoFormer"


def nm(m):
    return f"**{LABEL}**" if m == OURS else m


def main():
    global s_all
    s = pd.read_csv(os.path.join(ROOT, "analysis", "missing_robustness_sub.csv"))
    f = pd.read_csv(os.path.join(ROOT, "analysis",
                                 "missing_robustness_fulltest.csv"))
    s_all = s
    L = []
    W = L.append
    W("# Robustness to missing inputs (Reviewer 1, Minor #7)\n")
    W("Produced by [analysis/missing_robustness.py](analysis/missing_robustness.py) "
      "and [analysis/missing_report.py](analysis/missing_report.py). Inference "
      "only, on the existing checkpoints; no model was retrained.\n")

    W("## How the gaps are made and filled\n")
    W("The mechanism already in the codebase is used unchanged "
      "(`MeteoWindowDataset.missing_rate`, "
      "[src/data/dataset.py](src/data/dataset.py)):\n")
    W("1. A Bernoulli(`missing_rate`) mask is drawn **independently per cell** "
      "of the (96, n_channels) input window — per timestep *and* channel, not "
      "per whole timestep.")
    W("2. Masked cells become NaN and are refilled by **linear interpolation "
      "along time within that window**, per channel, over the surviving "
      "indices. A channel that is entirely missing is set to 0.0, the mean in "
      "scaled space.")
    W("3. The mask touches the **scaled input only**. Targets are never masked, "
      "so the metric stays comparable to the clean run.")
    W("4. The RNG is seeded from the run's seed, so the gap pattern is "
      "reproducible and identical across models at a given seed.\n")
    W("Budget: **5 seeds**, four rates, both datasets — the same seed budget as "
      "the main tables, so the numbers here are not on a smaller budget than "
      "the ones they sit next to.\n")

    n_w = s.groupby("dataset")[["n_windows", "test_stride"]].first()
    W("## Test subsample and its validation\n")
    W("The quantity of interest is a within-model difference between rates, so "
      "a systematic 1-in-k subsample of the test windows is used, with **all "
      "four rates computed on the same subsample** — the deltas are therefore "
      "exact by construction, not approximated.\n")
    W("| Dataset | full test windows | used | stride |")
    W("|---|---|---|---|")
    for ds, r in n_w.iterrows():
        full = {"jena": 13908, "beijing_aotizhongxin": 6895}[ds]
        W(f"| {ds} | {full} | {int(r.n_windows)} | 1-in-{int(r.test_stride)} |")
    W("")
    m = f.merge(s, on=["dataset", "model", "seed", "missing_rate"],
                suffixes=("_full", "_sub"))
    if len(m):
        d1 = m.MAE_sub - m.MAE_full
        d2 = m.RMSE_sub - m.RMSE_full
        W(f"Validated against **{len(m)} full-test runs**, paired exactly on "
          f"(dataset, model, seed, rate) — not compared as means:\n")
        W("| | mean difference | max abs | max relative |")
        W("|---|---|---|---|")
        W(f"| MAE | {d1.mean():+.4f} | {d1.abs().max():.4f} | "
          f"{100*(d1.abs()/m.MAE_full).max():.2f}% |")
        W(f"| RMSE | {d2.mean():+.4f} | {d2.abs().max():.4f} | "
          f"{100*(d2.abs()/m.RMSE_full).max():.2f}% |")
        W("")
        for src, name in ((f, "full test"), (s, "subsample")):
            k = src[src.dataset == "jena"].pivot_table(
                index=["model", "seed"], columns="missing_rate",
                values="MAE").dropna()
            if 0.0 in k and 0.20 in k:
                deg = 100 * (k[0.20] - k[0.0]) / k[0.0]
                W(f"- mean degradation at 20 % on Jena, **{name}** "
                  f"(n={len(deg)}): **{deg.mean():.3f}%**")
        W("\nThe subsample does not shift the measured degradation.\n")

    for ds, title in (("jena", "Jena"),
                      ("beijing_aotizhongxin", "Beijing (Aotizhongxin)")):
        d = s[s.dataset == ds]
        if d.empty:
            continue
        W(f"\n---\n\n## {title}\n")
        for met in ("MAE", "RMSE"):
            W(f"### {met} by missing rate (mean ± sd over 5 seeds)\n")
            W("| Model | " + " | ".join(f"{int(r*100)} %" for r in RATES)
              + " | Δ% at 20 % |")
            W("|---|" + "---|" * (len(RATES) + 1))
            rows = []
            for mdl, g in d.groupby("model"):
                cells, base, last = [], None, None
                for r in RATES:
                    v = g[g.missing_rate == r][met]
                    cells.append(f"{v.mean():.3f}±{v.std(ddof=1):.3f}")
                    if r == 0.0:
                        base = v.mean()
                    if r == 0.20:
                        last = v.mean()
                rows.append((100 * (last - base) / base, mdl, cells))
            for dpc, mdl, cells in sorted(rows):
                W(f"| {nm(mdl)} | " + " | ".join(cells) + f" | {dpc:+.2f} |")
            W("")

        W("### Per-target degradation, Δ% of MAE at 20 % missing\n")
        W("| Model | " + " | ".join(NAMES) + " |")
        W("|---|" + "---|" * 4)
        for mdl, g in sorted(d.groupby("model")):
            cs = []
            for n in NAMES:
                b = g[g.missing_rate == 0.0][f"MAE_{n}"].mean()
                e = g[g.missing_rate == 0.20][f"MAE_{n}"].mean()
                cs.append(f"{100*(e-b)/b:+.2f}")
            W(f"| {nm(mdl)} | " + " | ".join(cs) + " |")
        W("")

        rank0 = d[d.missing_rate == 0.0].groupby("model").MAE.mean().sort_values()
        rank2 = d[d.missing_rate == 0.20].groupby("model").MAE.mean().sort_values()
        W("### Does the ranking survive?\n")
        W("| Rank | clean (0 %) | 20 % missing |")
        W("|---|---|---|")
        for i, (a, b) in enumerate(zip(rank0.index, rank2.index), 1):
            mark = "" if a == b else "  ←changed"
            W(f"| {i} | {nm(a)} | {nm(b)}{mark} |")
        W("")
        same = list(rank0.index) == list(rank2.index)
        W(f"Ranking by MAE is **{'unchanged' if same else 'changed'}** between "
          f"the clean test and 20 % missing.\n")

    W("\n---\n\n## Verdict\n")
    jm = s[s.dataset == "jena"]
    piv = jm.pivot_table(index="model", columns="missing_rate", values="MAE")
    deg = (100 * (piv[0.20] - piv[0.0]) / piv[0.0]).sort_values()
    W(f"**The effect is small for every model — at most "
      f"{deg.max():.1f}% MAE for 20% of input cells destroyed.** That is a "
      f"property of the corruption scheme, not evidence that the models are "
      f"robust, and it must be written that way.\n")
    W("The mask is per-cell and is immediately repaired by linear "
      "interpolation along time. Hourly meteorological series are close to "
      "linear over a few hours, so an isolated missing cell is recovered almost "
      "exactly. The experiment therefore measures how good linear interpolation "
      "is on this data, with the model's sensitivity as a second-order effect. "
      "**Do not claim \"our model is robust to missing data\" on this basis.**\n")
    W(f"Our model degrades by {deg.get(OURS, float('nan')):.2f}% on Jena, which "
      f"is in the **lower half** of the field — Transformer, LSTM, Informer, "
      f"Autoformer and TimesNet all degrade less. So the claim to make is not "
      f"about robustness.\n")
    W("What does hold is **persistence of the first place**. Our model is "
      "first by MAE at every gap level on both datasets, and the margin over "
      "the runner-up does not shrink:\n")
    W("| Dataset | 0 % | 5 % | 10 % | 20 % |")
    W("|---|---|---|---|---|")
    for ds, t in (("jena", "Jena"), ("beijing_aotizhongxin", "Beijing")):
        dd = s_all[s_all.dataset == ds]
        cs = []
        for r in RATES:
            rk = dd[dd.missing_rate == r].groupby("model").MAE.mean().sort_values()
            cs.append(f"+{rk.iloc[1]-rk.iloc[0]:.4f} vs {rk.index[1]}")
        W(f"| {t} | " + " | ".join(cs) + " |")
    W("")
    W("The **rest** of the ranking is not stable: mid-field positions swap "
      "between the clean test and 20 % missing on both datasets (e.g. on Jena "
      "iTransformer/TimesNet and DLinear/Transformer exchange places). Those "
      "pairs are separated by less than their own seed sd, so the swaps carry "
      "no signal — but the paper should say \"first place is unaffected\", "
      "not \"the ranking is unaffected\", because the latter is checkably "
      "false.\n")
    W("If a stronger robustness claim is wanted, the corruption scheme has to "
      "be harder and closer to an operational failure — whole timesteps "
      "dropped, or a contiguous block of hours lost, or one sensor offline for "
      "the entire window. Those are cheap to add (the same inference harness) "
      "and would be a better answer to the reviewer than the present result. "
      "They are **not** included here, and nothing above should be read as "
      "covering them.\n")

    dst = os.path.join(ROOT, "analysis", "missing_robustness.md")
    open(dst, "w").write("\n".join(L))
    print("->", dst)


if __name__ == "__main__":
    main()
