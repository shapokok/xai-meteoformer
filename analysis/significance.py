"""Significance testing for the forecast comparison.

Three families, all paired, replacing the unpaired Welch t-test that
src/report.py used to emit:

1. Diebold-Mariano on the loss differential per test window, with a
   Newey-West HAC variance and automatic lag selection. Run separately
   per target channel and on the channel-averaged aggregate, for
   absolute and squared loss. This is the test that has power: n is the
   number of test windows, and it accounts for the serial correlation
   that overlapping forecast windows induce.
2. Paired t-test and paired Wilcoxon signed-rank over the 5 seed means,
   which is the seed-level analogue of what the paper reported.
3. Holm correction inside the family "proposed model vs each baseline",
   applied per metric.

The DM loss differential is built from the per-window loss of each seed
averaged over the 5 seeds -- NOT from seed-averaged predictions. Averaging
predictions would build a 5-member ensemble, which is a different and
stronger model than the one the paper reports.

Test data is read, never fitted.

Outputs -> analysis/significance.md
        -> analysis/significance_dm.csv
        -> paper/tables/significance_{dataset}.tex
"""

import os

import numpy as np
import pandas as pd
import sys
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "predictions")
NAMES = ["T", "RH", "P", "WS"]
DATASETS = ["jena", "beijing_aotizhongxin"]
OURS = ("XAI-MeteoFormer", "no_revin")
LABEL = "MeteoFormer"
SEEDS = range(5)

# Baseline normalization: "selected" (main text; the variant with the lower
# mean validation loss, analysis/selection.py) or "historical" (appendix).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selection import selected_suffix  # noqa: E402
MODE = "historical" if "--historical" in sys.argv else "selected"
TAG = "" if MODE == "selected" else "_historical"
NORM_NOTE = ("each in the normalization variant with the lower mean validation "
             "loss (analysis/selection.py)." if MODE == "selected" else
             "each in its historical normalization (appendix).")


def bl_abl(model, ds):
    """ablation part of a baseline's tag, including the normalization suffix"""
    return "full" + (selected_suffix(model, ds) if MODE == "selected" else "")


# --------------------------------------------------------------------------- #
def newey_west_lag(n: int) -> int:
    """Newey & West (1994) automatic bandwidth, the standard plug-in rule."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def hac_var(d: np.ndarray, lag: int) -> float:
    """Newey-West long-run variance of the mean of d."""
    n = len(d)
    dm = d - d.mean()
    # Apple's Accelerate BLAS raises spurious FP flags on this machine; the
    # dot products themselves are correct, verified against an explicit sum.
    with np.errstate(all="ignore"):
        g0 = float(dm @ dm) / n
        s = g0
        for k in range(1, lag + 1):
            gk = float(dm[k:] @ dm[:-k]) / n
            s += 2.0 * (1.0 - k / (lag + 1.0)) * gk   # Bartlett kernel
    return s


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray):
    """d = loss_a - loss_b, one value per test window. Negative favours a.

    Returns (mean d, DM stat, two-sided p, lag). Uses the Harvey-
    Leybourne-Newbold small-sample correction and a t reference with
    n-1 df, which is the recommended practice for h-step forecasts.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    lag = max(newey_west_lag(n), 1)
    v = hac_var(d, lag)
    if v <= 0:
        return float(d.mean()), np.nan, np.nan, lag
    dm = d.mean() / np.sqrt(v / n)
    # Harvey, Leybourne & Newbold (1997) correction for horizon h = lag+1
    h = lag + 1
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_c = dm * corr
    p = 2 * (1 - stats.t.cdf(abs(dm_c), df=n - 1))
    return float(d.mean()), float(dm_c), float(p), lag


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    adj = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j])
        adj[j] = min(run, 1.0)
    return adj


# --------------------------------------------------------------------------- #
def load(model, ablation, ds):
    """list of per-seed test predictions, each (N, H, C)"""
    ps = []
    for s in SEEDS:
        f = os.path.join(PRED, f"{model}_{ds}_{ablation}_s{s}_pred.npy")
        if os.path.exists(f):
            ps.append(np.load(f).astype(np.float64))
    return (ps, len(ps)) if ps else (None, 0)


def per_seed_metrics(model, ablation, ds, true):
    """MAE / RMSE / R2 for each seed separately"""
    out = []
    for s in SEEDS:
        f = os.path.join(PRED, f"{model}_{ds}_{ablation}_s{s}_pred.npy")
        if not os.path.exists(f):
            continue
        p = np.load(f).astype(np.float64)
        e = p - true
        ss_res = (e ** 2).sum(axis=(0, 1))
        ss_tot = ((true - true.mean(axis=(0, 1), keepdims=True)) ** 2).sum(axis=(0, 1))
        out.append({"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
                    "R2": np.mean(1 - ss_res / np.clip(ss_tot, 1e-9, None))})
    return pd.DataFrame(out)


def window_loss(preds, true, kind, ch=None):
    """one loss value per test window, averaged over seeds.

    The seed average is taken on the LOSS, not on the prediction, so the
    statistic describes a typical single run rather than an ensemble.
    """
    out = []
    for pred in preds:
        e = pred - true if ch is None else pred[:, :, ch] - true[:, :, ch]
        ax = (1, 2) if ch is None else 1
        out.append(np.abs(e).mean(axis=ax) if kind == "abs"
                   else (e ** 2).mean(axis=ax))
    return np.mean(out, axis=0)


def write_tex_tables(dm):
    """Emit the MDPI tables that replace the old Welch t-test ones."""
    out = os.path.join(ROOT, "paper", "tables")
    os.makedirs(out, exist_ok=True)
    for ds in DATASETS:
        d = dm[dm.dataset == ds]
        if d.empty:
            continue
        bl = sorted(d.baseline.unique())

        def cell(b, loss, ch):
            r = d[(d.baseline == b) & (d.loss == loss) & (d.channel == ch)]
            if r.empty or pd.isna(r.p_holm.iloc[0]):
                return "--", "--"
            v, pv = r.mean_diff.iloc[0], r.p_holm.iloc[0]
            star = "$^{*}$" if pv < 0.05 else ""
            ps = "$<$0.001" if pv < 1e-3 else f"{pv:.3f}"
            return f"{v:+.3f}{star}", ps

        lines = ["\\begin{tabular}{lcccc}", "\\toprule",
                 "Baseline & $\\Delta L_1$ & $p_{\\mathrm{Holm}}$ & "
                 "$\\Delta L_2$ & $p_{\\mathrm{Holm}}$ \\\\", "\\midrule"]
        for b in bl:
            a1, p1 = cell(b, "abs", "agg")
            a2, p2 = cell(b, "sq", "agg")
            lines.append(f"{b} & {a1} & {p1} & {a2} & {p2} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write_tex(os.path.join(out, f"significance{TAG}_{ds}.tex"), "\n".join(lines),
                  f"Diebold--Mariano test of {LABEL} against each baseline on "
                  f"{ds}, on the per-window loss differential with a "
                  f"Newey--West HAC variance (automatic lag) and the "
                  f"Harvey--Leybourne--Newbold correction. $L_1$ = absolute "
                  f"loss, $L_2$ = squared loss; the per-window loss is averaged "
                  f"over the 5 seeds. Negative $\\Delta$ favours {LABEL}. "
                  f"$^{{*}}$: $p<0.05$ after Holm correction within each "
                  f"column. Baselines: {NORM_NOTE}", f"tab:sig{TAG}_{ds}")

        lines = ["\\begin{tabular}{lcccccccc}", "\\toprule",
                 "& \\multicolumn{4}{c}{$\\Delta L_1$} & "
                 "\\multicolumn{4}{c}{$\\Delta L_2$} \\\\",
                 "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
                 "Baseline & " + " & ".join(NAMES) + " & " +
                 " & ".join(NAMES) + " \\\\", "\\midrule"]
        for b in bl:
            cs = [cell(b, "abs", n)[0] for n in NAMES] + \
                 [cell(b, "sq", n)[0] for n in NAMES]
            lines.append(f"{b} & " + " & ".join(cs) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write_tex(os.path.join(out, f"significance_channels{TAG}_{ds}.tex"),
                  "\n".join(lines),
                  f"Diebold--Mariano differential per target channel on {ds}. "
                  f"Negative favours {LABEL}. $^{{*}}$: $p<0.05$ after Holm "
                  f"correction within each column. The sign flips between "
                  f"temperature and relative humidity.",
                  f"tab:sig_ch{TAG}_{ds}")


def write_tex(path, body, caption, label):
    with open(path, "w") as f:
        f.write("\\begin{table}[H]\n\\caption{" + caption + "}\n")
        f.write("\\label{" + label + "}\n" + body + "\n\\end{table}\n")
    print("  wrote", path)


def main():
    dm_rows, L = [], []
    W = L.append
    W("# Significance testing\n")
    W("Produced by [analysis/significance.py](analysis/significance.py). "
      "Test predictions are read; nothing is fitted on test.\n")
    W("Replaces the unpaired Welch $t$-test previously emitted into "
      "`paper/tables/significance_*.tex`. That test was **unpaired** "
      "(`scipy.stats.ttest_ind`) although the seeds are matched across models, "
      "which is both less powerful and not the right null.\n")
    W("**Negative Δ favours " + LABEL + " throughout.**\n")
    W(f"**Baselines: {NORM_NOTE}** Run with `--historical` for the appendix "
      "version against every baseline's historical normalization.\n")

    for ds in DATASETS:
        true_f = os.path.join(PRED, f"{ds}_true.npy")
        if not os.path.exists(true_f):
            W(f"\n_{ds}: ground truth missing_\n")
            continue
        true = np.load(true_f).astype(np.float64)
        ours, n_ours = load(*OURS, ds)
        if ours is None:
            continue
        baselines = sorted({f.split(f"_{ds}_")[0] for f in os.listdir(PRED)
                            if f.endswith("_pred.npy") and f"_{ds}_full_" in f
                            and not f.startswith(OURS[0])})

        title = "Jena" if ds == "jena" else "Beijing (Aotizhongxin)"
        W(f"\n---\n\n## {title}\n")

        # ---------------- 1. Diebold-Mariano ---------------- #
        W("### 1. Diebold–Mariano, per test window\n")
        W(f"Loss differential per test window (n = {len(true)}), Newey–West HAC "
          f"variance with the automatic lag rule "
          f"$\\lfloor 4(n/100)^{{2/9}}\\rfloor$ = {newey_west_lag(len(true))}, "
          "Harvey–Leybourne–Newbold small-sample correction, two-sided. "
          "Holm correction within each (metric, channel) family of "
          f"{len(baselines)} baselines. The differential is the per-window "
          "loss averaged over the 5 seeds, so it describes a typical single "
          "run, not a seed ensemble.\n")

        for kind, kname in (("abs", "absolute loss (MAE)"),
                            ("sq", "squared loss (MSE)")):
            W(f"#### {kname}\n")
            W("| Baseline | " + " | ".join(
                ["aggregate"] + NAMES) + " |")
            W("|---|" + "---|" * 5)
            cell, praw = {}, {c: [] for c in ["agg"] + NAMES}
            for b in baselines:
                bp, _ = load(b, bl_abl(b, ds), ds)
                cell[b] = {}
                for cname, ch in [("agg", None)] + list(zip(NAMES, range(4))):
                    la = window_loss(ours, true, kind, ch)
                    lb = window_loss(bp, true, kind, ch)
                    mean_d, stat, p, lag = diebold_mariano(la, lb)
                    cell[b][cname] = (mean_d, stat, p)
                    praw[cname].append(p)
                    dm_rows.append({"dataset": ds, "loss": kind, "channel": cname,
                                    "baseline": b, "mean_diff": mean_d,
                                    "DM": stat, "p_raw": p, "lag": lag})
            adj = {c: holm(praw[c]) for c in praw}
            for i, b in enumerate(baselines):
                cs = []
                for cname in ["agg"] + NAMES:
                    md, st, _ = cell[b][cname]
                    pa = adj[cname][i]
                    star = "**\\***" if pa < 0.05 else ""
                    cs.append(f"{md:+.4f} (p={pa:.3g}){star}")
                    for r in dm_rows:
                        if (r["dataset"] == ds and r["loss"] == kind
                                and r["channel"] == cname and r["baseline"] == b):
                            r["p_holm"] = pa
                W(f"| {b} | " + " | ".join(cs) + " |")
            W("")
            W("\\* = significant at p<0.05 after Holm.\n")

        # ---------------- 2. seed-level paired tests ---------------- #
        W("### 2. Paired tests over the 5 seed means\n")
        W("Both tests pair seed $i$ of our model with seed $i$ of the baseline. "
          "**With n=5 the smallest attainable two-sided p for the Wilcoxon "
          "signed-rank test is 0.0625**, so it can never reach p<0.05 no matter "
          "how large the effect. It is reported for completeness only; the "
          "Diebold–Mariano test above is the one with power.\n")
        om = per_seed_metrics(*OURS, ds, true)
        for met in ("MAE", "RMSE", "R2"):
            W(f"#### {met}\n")
            W("| Baseline | ours | baseline | Δ | paired t p | Holm | Wilcoxon p |")
            W("|---|---|---|---|---|---|---|")
            rows, ps = [], []
            for b in baselines:
                bm = per_seed_metrics(b, bl_abl(b, ds), ds, true)
                n = min(len(om), len(bm))
                x, y = om[met].values[:n], bm[met].values[:n]
                tp = stats.ttest_rel(x, y).pvalue
                try:
                    wp = stats.wilcoxon(x, y).pvalue
                except ValueError:
                    wp = np.nan
                rows.append((b, x.mean(), y.mean(), (x - y).mean(), tp, wp))
                ps.append(tp)
            adj = holm(ps)
            for (b, xm, ym, dd, tp, wp), pa in zip(rows, adj):
                star = "**\\***" if pa < 0.05 else ""
                W(f"| {b} | {xm:.4f} | {ym:.4f} | {dd:+.4f} | {tp:.4g} | "
                  f"{pa:.4g}{star} | {wp:.4g} |")
            W("")

    write_tex_tables(pd.DataFrame(dm_rows))
    pd.DataFrame(dm_rows).to_csv(
        os.path.join(ROOT, "analysis", f"significance{TAG}_dm.csv"), index=False)
    open(os.path.join(ROOT, "analysis", f"significance{TAG}.md"), "w").write("\n".join(L))
    print(f"-> analysis/significance{TAG}.md, analysis/significance{TAG}_dm.csv")
    return pd.DataFrame(dm_rows)


if __name__ == "__main__":
    main()
