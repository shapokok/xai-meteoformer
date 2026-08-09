"""Build every table and figure for the paper from saved artefacts.

    python src/report.py --results results.csv --results2 beijing_results.csv \
        --cls cls_metrics.csv --xai xai_metrics.csv --out paper/

Reads only files that already exist; anything missing is skipped with a note
rather than aborting, so this can be run while experiments are still coming in.

Outputs
  paper/tables/*.tex      MDPI-ready booktabs tables
  paper/figures/*.pdf     300 dpi, vector
  paper/summary.txt       the numbers to quote in the text
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})

PROPOSED = "XAI-MeteoFormer"
LABEL = "MeteoFormer"          # name used in the paper
SHIFT_HEAD = ["params", "train_time_s", "infer_time_s", "best_val", "epochs_run",
              "seq_len", "pred_len", "d_model", "n_layers", "lr", "lambda_ent"]
SHIFT_ACT = ["lambda_ent", "params", "train_time_s", "infer_time_s", "best_val",
             "epochs_run", "seq_len", "pred_len", "d_model", "n_layers", "lr"]


# --------------------------------------------------------------------------- #
def load_results(paths):
    """Merge result CSVs, repairing the column shift left by the old writer."""
    frames = []
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        if set(SHIFT_HEAD) <= set(d.columns):
            d[SHIFT_HEAD] = d[SHIFT_HEAD].astype(float)
            sh = d["params"] < 1000          # lambda_ent sitting in params
            if sh.any():
                d.loc[sh, SHIFT_ACT] = d.loc[sh, SHIFT_HEAD].values
                print(f"  {os.path.basename(p)}: repaired {int(sh.sum())} rows")
        frames.append(d)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    # duplicates arise when a resume key failed to match; keep the last write
    before = len(d)
    d = d.drop_duplicates(["model", "dataset", "ablation", "seed"], keep="last")
    if len(d) < before:
        print(f"  dropped {before - len(d)} duplicate runs")
    return d


def display_name(row):
    if row["model"] != PROPOSED:
        return row["model"]
    return LABEL if row["ablation"] == "no_revin" else f"{LABEL} ({row['ablation']})"


def fmt(m, s, digits=3):
    return f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"


def write_tex(path, body, caption, label):
    with open(path, "w") as f:
        f.write("\\begin{table}[H]\n\\caption{" + caption + "}\n")
        f.write("\\label{" + label + "}\n")
        f.write(body)
        f.write("\n\\end{table}\n")
    print("  wrote", path)


# --------------------------------------------------------------------------- #
def table_main(d, out, summary):
    """Accuracy per dataset, proposed model vs every baseline."""
    for ds in sorted(d.dataset.unique()):
        sub = d[(d.dataset == ds) & (d.ablation.isin(["full", "no_revin"]))].copy()
        sub = sub[~((sub.model == PROPOSED) & (sub.ablation == "full"))]
        sub["name"] = sub.apply(display_name, axis=1)
        g = sub.groupby("name")[["MAE", "RMSE", "R2"]].agg(["mean", "std"])
        g = g.sort_values(("MAE", "mean"))

        lines = ["\\begin{tabular}{lccc}", "\\toprule",
                 "Model & MAE & RMSE & $R^2$ \\\\", "\\midrule"]
        best = {m: (g[(m, "mean")].min() if m != "R2" else g[(m, "mean")].max())
                for m in ["MAE", "RMSE", "R2"]}
        for name, r in g.iterrows():
            cells = []
            for m in ["MAE", "RMSE", "R2"]:
                v = fmt(r[(m, "mean")], r[(m, "std")])
                if abs(r[(m, "mean")] - best[m]) < 1e-9:
                    v = "\\textbf{" + v + "}"
                cells.append(v)
            nm = "\\textbf{" + name + "}" if name == LABEL else name
            lines += [f"{nm} & " + " & ".join(cells) + " \\\\"]
        lines += ["\\bottomrule", "\\end{tabular}"]
        write_tex(os.path.join(out, "tables", f"main_{ds}.tex"), "\n".join(lines),
                  f"Forecasting accuracy on {ds}, mean $\\pm$ s.d. over 5 seeds. "
                  f"Best per column in bold.", f"tab:main_{ds}")
        summary.append(f"\n[{ds}] accuracy\n" + g.round(4).to_string())


def table_significance(d, out, summary):
    """Welch t-test of the proposed model against each baseline, Holm-corrected."""
    for ds in sorted(d.dataset.unique()):
        ours = d[(d.model == PROPOSED) & (d.ablation == "no_revin") &
                 (d.dataset == ds)]
        if ours.empty:
            continue
        recs = []
        for m in sorted(d[(d.dataset == ds) & (d.ablation == "full")].model.unique()):
            if m == PROPOSED:
                continue
            b = d[(d.model == m) & (d.ablation == "full") & (d.dataset == ds)]
            row = {"baseline": m}
            for met in ["MAE", "RMSE", "R2"]:
                t, p = stats.ttest_ind(ours[met], b[met], equal_var=False)
                diff = ours[met].mean() - b[met].mean()
                row[f"{met}_d"] = diff
                row[f"{met}_p"] = p
            recs.append(row)
        r = pd.DataFrame(recs)
        # Holm within each metric family
        for met in ["MAE", "RMSE", "R2"]:
            p = r[f"{met}_p"].values
            order = np.argsort(p)
            m_ = len(p)
            adj = np.empty(m_)
            run = 0.0
            for i, idx in enumerate(order):
                run = max(run, (m_ - i) * p[idx])
                adj[idx] = min(run, 1.0)
            r[f"{met}_holm"] = adj

        lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
                 "Baseline & $\\Delta$MAE & $p_{\\mathrm{Holm}}$ & $\\Delta$RMSE & "
                 "$p_{\\mathrm{Holm}}$ & $\\Delta R^2$ & $p_{\\mathrm{Holm}}$ \\\\",
                 "\\midrule"]
        for _, x in r.iterrows():
            c = []
            for met in ["MAE", "RMSE", "R2"]:
                pv = x[f"{met}_holm"]
                star = "$^{*}$" if pv < 0.05 else ""
                c += [f"{x[f'{met}_d']:+.3f}{star}", f"{pv:.3f}"]
            lines.append(f"{x['baseline']} & " + " & ".join(c) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write_tex(os.path.join(out, "tables", f"significance_{ds}.tex"),
                  "\n".join(lines),
                  f"Welch $t$-test of {LABEL} against each baseline on {ds}. "
                  f"Negative $\\Delta$MAE favours {LABEL}. $^{{*}}$: $p<0.05$ "
                  f"after Holm correction within each metric.",
                  f"tab:sig_{ds}")
        summary.append(f"\n[{ds}] significance\n" + r.round(4).to_string(index=False))


def table_ablation(d, out, summary):
    sub = d[(d.model == PROPOSED) & (d.dataset == "jena")]
    if sub.ablation.nunique() < 3:
        print("  ablation: not enough variants, skipped")
        return
    g = (sub.groupby("ablation")[["best_val", "MAE", "RMSE", "R2"]]
            .agg(["mean", "std"]).sort_values(("best_val", "mean")))
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Variant & Val.\\ loss & MAE & RMSE & $R^2$ \\\\", "\\midrule"]
    for name, r in g.iterrows():
        lines.append(
            f"{name.replace('_', ' ')} & {r[('best_val','mean')]:.4f} & "
            + " & ".join(fmt(r[(m, 'mean')], r[(m, 'std')])
                         for m in ["MAE", "RMSE", "R2"]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(out, "tables", "ablation.tex"), "\n".join(lines),
              "Ablation on Jena, ordered by validation loss. The architecture "
              "was selected on validation, not on test.", "tab:ablation")
    summary.append("\n[ablation, jena]\n" + g.round(4).to_string())


def table_fidelity(x, out, summary):
    """Every model scored with the same permutation ranking, plus the proposed
    model's own attention ranking as a separate row."""
    for ds in sorted(x.dataset.unique()):
        sub = x[(x.dataset == ds) & (x.exclude_time == True)]
        if "fidelity_gain_perm_rel" not in sub.columns:
            print("  fidelity: new columns missing, skipped")
            return
        sub = sub[sub["fidelity_gain_perm_rel"].notna()]
        rows = []
        ours = sub[sub.ablation == "no_revin"]
        if not ours.empty:
            rows.append({"Model": f"\\textbf{{{LABEL}}} (permutation)",
                         "fid": ours["fidelity_gain_perm_rel"].mean(),
                         "sd": ours["fidelity_gain_perm_rel"].std()})
            rows.append({"Model": f"\\textbf{{{LABEL}}} (built-in attention)",
                         "fid": ours["fidelity_gain_attn_rel"].mean(),
                         "sd": ours["fidelity_gain_attn_rel"].std()})
        for _, r in sub[sub.model != PROPOSED].iterrows():
            rows.append({"Model": r.model,
                         "fid": r["fidelity_gain_perm_rel"], "sd": np.nan})
        t = pd.DataFrame(rows).sort_values("fid", ascending=False)
        lines = ["\\begin{tabular}{lc}", "\\toprule",
                 "Model (ranking) & Fidelity gain \\\\", "\\midrule"]
        for _, r in t.iterrows():
            v = (f"{r['fid']:.3f} $\\pm$ {r['sd']:.3f}"
                 if not np.isnan(r["sd"]) else f"{r['fid']:.3f}")
            lines.append(f"{r['Model']} & {v} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write_tex(os.path.join(out, "tables", f"fidelity_{ds}.tex"), "\n".join(lines),
                  f"Explanation fidelity on {ds}, physical channels only. All "
                  f"models are ranked by permutation importance so the numbers "
                  f"are comparable; the built-in attention of {LABEL} is listed "
                  f"separately.", f"tab:fid_{ds}")
        summary.append(f"\n[{ds}] fidelity\n" + t.round(3).to_string(index=False))


def table_cls(c, out, summary):
    for ds in sorted(c.dataset.unique()):
        g = (c[c.dataset == ds].groupby("model")[["AUC", "AP", "F1",
                                                  "Precision", "Recall"]]
             .agg(["mean", "std"]).sort_values(("AP", "mean"), ascending=False))
        lines = ["\\begin{tabular}{lccccc}", "\\toprule",
                 "Model & AUC & AP & F1 & Precision & Recall \\\\", "\\midrule"]
        for name, r in g.iterrows():
            lines.append(f"{name} & " + " & ".join(
                fmt(r[(m, "mean")], r[(m, "std")])
                for m in ["AUC", "AP", "F1", "Precision", "Recall"]) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write_tex(os.path.join(out, "tables", f"events_{ds}.tex"), "\n".join(lines),
                  f"Frost-event detection on {ds}. Every model is scored the same "
                  f"way: the ranking score is the negated predicted temperature "
                  f"and the decision threshold is the freezing point, so nothing "
                  f"is tuned.", f"tab:events_{ds}")
        summary.append(f"\n[{ds}] events\n" + g.round(4).to_string())


# --------------------------------------------------------------------------- #
def fig_accuracy_vs_fidelity(d, x, out):
    """The paper's headline figure: accuracy and explanation fidelity are
    unrelated across models."""
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, ds in zip(axes, sorted(d.dataset.unique())):
        sub = x[(x.dataset == ds) & (x.exclude_time == True) &
                x["fidelity_gain_perm_rel"].notna()]
        pts = []
        for _, r in sub.iterrows():
            if r.model == PROPOSED and r.ablation != "no_revin":
                continue
            acc = d[(d.dataset == ds) & (d.model == r.model) &
                    (d.ablation == ("no_revin" if r.model == PROPOSED else "full"))]
            if acc.empty:
                continue
            pts.append((acc.MAE.mean(),
                        r["fidelity_gain_perm_rel"],
                        LABEL if r.model == PROPOSED else r.model))
        if not pts:
            continue
        pts = pd.DataFrame(pts, columns=["mae", "fid", "name"]).groupby(
            "name", as_index=False).mean()
        for _, p in pts.iterrows():
            mine = p["name"] == LABEL
            ax.scatter(p.mae, p.fid, s=70 if mine else 40,
                       marker="*" if mine else "o",
                       zorder=3 if mine else 2,
                       color="#c0392b" if mine else "#34495e")
            ax.annotate(p["name"], (p.mae, p.fid), fontsize=6,
                        xytext=(4, 3), textcoords="offset points")
        if len(pts) > 2:
            rho = stats.spearmanr(pts.mae, pts.fid).correlation
            ax.set_title(f"{ds}  ($\\rho$ = {rho:.2f})", fontsize=9)
        ax.set_xlabel("MAE (lower is better)")
        ax.set_ylabel("Fidelity gain")
    p = os.path.join(out, "figures", "accuracy_vs_fidelity.pdf")
    fig.savefig(p); plt.close(fig); print("  wrote", p)


def fig_fidelity_curves(xai_dir, out):
    """Error rise as the top-k channels are permuted, per ranking method."""
    files = sorted(glob.glob(os.path.join(xai_dir, "*no_revin*_xai.npz")))
    if not files:
        print("  fidelity curves: no npz found, skipped")
        return
    by_ds = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        ds = "beijing" if "beijing" in f else "jena"
        by_ds.setdefault(ds, []).append(z)
    fig, axes = plt.subplots(1, len(by_ds), figsize=(3.7 * len(by_ds), 3.2),
                             squeeze=False)
    for ax, (ds, zs) in zip(axes[0], sorted(by_ds.items())):
        ks = zs[0]["fidelity_ks"]
        for key, lab, st in [("fidelity_random", "random", ":"),
                             ("fidelity_attn", "built-in attention", "--"),
                             ("fidelity_perm", "permutation", "-"),
                             ("fidelity_shap", "SHAP", "-.")]:
            cur = [z[key] for z in zs if key in z]
            if not cur:
                continue
            m = np.mean(cur, axis=0)
            s = np.std(cur, axis=0)
            ax.plot(ks, m, st, label=lab, lw=1.6)
            ax.fill_between(ks, m - s, m + s, alpha=0.15)
        ax.set_xlabel("channels permuted (k)")
        ax.set_ylabel("MAE (normalised units)")
        ax.set_title(ds, fontsize=9)
        ax.legend(fontsize=7)
    p = os.path.join(out, "figures", "fidelity_curves.pdf")
    fig.savefig(p); plt.close(fig); print("  wrote", p)


def fig_importance_heatmap(xai_dir, out):
    """Variable attention, targets x channels, averaged over seeds."""
    for ds in ["jena", "beijing"]:
        files = sorted(glob.glob(os.path.join(xai_dir, f"*{ds}*no_revin*_xai.npz")))
        files = [f for f in files if "var_attn" in np.load(f, allow_pickle=True)]
        if not files:
            continue
        mats, chans = [], None
        for f in files:
            z = np.load(f, allow_pickle=True)
            mats.append(z["var_attn"].mean(axis=0))
            chans = [str(c) for c in z["channels"]]
        M = np.mean(mats, axis=0)
        fig, ax = plt.subplots(figsize=(max(5, 0.34 * len(chans)), 2.4))
        im = ax.imshow(M, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(chans)))
        ax.set_xticklabels(chans, rotation=90, fontsize=6)
        ax.set_yticks(range(M.shape[0]))
        ax.set_yticklabels(["T", "RH", "P", "WS"][:M.shape[0]], fontsize=7)
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="attention weight")
        p = os.path.join(out, "figures", f"variable_attention_{ds}.pdf")
        fig.savefig(p); plt.close(fig); print("  wrote", p)


def fig_horizon(d, out):
    cols = [c for c in ["MAE_h1", "MAE_h6", "MAE_h12", "MAE_h24"] if c in d.columns]
    if not cols:
        return
    hs = [int(c.split("h")[1]) for c in cols]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), squeeze=False)
    for ax, ds in zip(axes[0], sorted(d.dataset.unique())):
        sub = d[(d.dataset == ds) & (d.ablation.isin(["full", "no_revin"]))].copy()
        sub = sub[~((sub.model == PROPOSED) & (sub.ablation == "full"))]
        sub["name"] = sub.apply(display_name, axis=1)
        for name, g in sub.groupby("name"):
            mine = name == LABEL
            ax.plot(hs, [g[c].mean() for c in cols], marker="o", ms=3,
                    lw=2.0 if mine else 1.0, alpha=1.0 if mine else 0.55,
                    label=name, color="#c0392b" if mine else None, zorder=3 if mine else 2)
        ax.set_xlabel("forecast horizon (h)")
        ax.set_ylabel("MAE")
        ax.set_title(ds, fontsize=9)
    axes[0][-1].legend(fontsize=6, ncol=2)
    p = os.path.join(out, "figures", "error_vs_horizon.pdf")
    fig.savefig(p); plt.close(fig); print("  wrote", p)


def fig_events(c, out):
    if c is None or "TP" not in c.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), squeeze=False)
    for ax, ds in zip(axes[0], sorted(c.dataset.unique())):
        g = c[c.dataset == ds].groupby("model")[["TN", "FP", "FN", "TP"]].mean()
        g = g.loc[g.index.sort_values()]
        prec = g.TP / (g.TP + g.FP).clip(lower=1)
        rec = g.TP / (g.TP + g.FN).clip(lower=1)
        for name in g.index:
            mine = name == PROPOSED
            ax.scatter(rec[name], prec[name], s=70 if mine else 35,
                       marker="*" if mine else "o",
                       color="#c0392b" if mine else "#34495e", zorder=3)
            ax.annotate(LABEL if mine else name, (rec[name], prec[name]),
                        fontsize=6, xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel("recall"); ax.set_ylabel("precision")
        ax.set_title(f"frost events, {ds}", fontsize=9)
    p = os.path.join(out, "figures", "event_precision_recall.pdf")
    fig.savefig(p); plt.close(fig); print("  wrote", p)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", default=["results.csv"])
    ap.add_argument("--cls", default="cls_metrics.csv")
    ap.add_argument("--xai", default="xai_metrics.csv")
    ap.add_argument("--xai_dir", default="xai")
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    summary = []

    print("results:")
    d = load_results(args.results)
    if d is not None:
        table_main(d, args.out, summary)
        table_significance(d, args.out, summary)
        table_ablation(d, args.out, summary)
        fig_horizon(d, args.out)

    x = pd.read_csv(args.xai) if os.path.exists(args.xai) else None
    if x is not None:
        print("xai:")
        table_fidelity(x, args.out, summary)
        if d is not None:
            fig_accuracy_vs_fidelity(d, x, args.out)
    fig_fidelity_curves(args.xai_dir, args.out)
    fig_importance_heatmap(args.xai_dir, args.out)

    c = pd.read_csv(args.cls) if os.path.exists(args.cls) else None
    if c is not None:
        print("events:")
        table_cls(c, args.out, summary)
        fig_events(c, args.out)

    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write("\n".join(summary))
    print(f"\nsummary -> {args.out}/summary.txt")


if __name__ == "__main__":
    main()
