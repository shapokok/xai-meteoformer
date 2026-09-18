"""Per-target and per-horizon accuracy for every model (Reviewer 2).

Computed from the saved test predictions, mean +- sd over 5 seeds.
Emits markdown for analysis/ and MDPI tables for paper/tables/.
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "predictions")
NAMES = ["T", "RH", "P", "WS"]
UNITS = {"T": "$^\\circ$C", "RH": "\\%", "P": "mbar", "WS": "m\\,s$^{-1}$"}
HZ = [1, 6, 12, 24]
DATASETS = ["jena", "beijing_aotizhongxin"]
OURS = ("XAI-MeteoFormer", "no_revin")
LABEL = "MeteoFormer"
SEEDS = range(5)


def variants(ds):
    v = [OURS]
    for f in sorted(os.listdir(PRED)):
        if f.endswith("_pred.npy") and f"_{ds}_full_" in f:
            m = f.split(f"_{ds}_")[0]
            if m != OURS[0] and (m, "full") not in v:
                v.append((m, "full"))
    return v


def seeds_of(model, abl, ds):
    out = []
    for s in SEEDS:
        f = os.path.join(PRED, f"{model}_{ds}_{abl}_s{s}_pred.npy")
        if os.path.exists(f):
            out.append(np.load(f).astype(np.float64))
    return out


def stat(vals):
    v = np.asarray(vals, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def md(m, s):
    return f"{m:.3f}±{s:.3f}"


def tex(m, s):
    return f"{m:.3f} $\\pm$ {s:.3f}"


def write_tex(path, body, caption, label):
    with open(path, "w") as f:
        f.write("\\begin{table}[H]\n\\caption{" + caption + "}\n")
        f.write("\\label{" + label + "}\n" + body + "\n\\end{table}\n")
    print("  wrote", os.path.relpath(path, ROOT))


def main():
    L = []
    W = L.append
    W("# Per-target and per-horizon accuracy (Reviewer 2)\n")
    W("From the saved test predictions, mean±sd over 5 seeds, physical units "
      "(T °C, RH %, P mbar, WS m s⁻¹). Produced by "
      "[analysis/per_target_horizon.py](analysis/per_target_horizon.py).\n")
    W("The aggregate MAE/RMSE the paper reports is the unweighted mean over "
      "the four channels. Because RH errors are an order of magnitude larger "
      "than T errors in physical units, the aggregate is dominated by RH — "
      "which is exactly why the per-channel table below matters.\n")

    for ds in DATASETS:
        tf = os.path.join(PRED, f"{ds}_true.npy")
        if not os.path.exists(tf):
            W(f"\n_{ds}: ground truth missing_\n")
            continue
        true = np.load(tf).astype(np.float64)
        title = "Jena" if ds == "jena" else "Beijing (Aotizhongxin)"
        W(f"\n---\n\n## {title}\n")
        vs = variants(ds)

        # ---------- per target ----------
        for met in ("MAE", "RMSE", "R2"):
            W(f"### Per target — {met}\n")
            W("| Model | " + " | ".join(NAMES) + " |")
            W("|---|" + "---|" * 4)
            rows = []
            for model, abl in vs:
                lab = LABEL if (model, abl) == OURS else model
                ps = seeds_of(model, abl, ds)
                cells, raw = [], []
                for c in range(4):
                    vals = []
                    for p in ps:
                        e = p[:, :, c] - true[:, :, c]
                        if met == "MAE":
                            vals.append(np.abs(e).mean())
                        elif met == "RMSE":
                            vals.append(np.sqrt((e ** 2).mean()))
                        else:
                            sst = ((true[:, :, c] - true[:, :, c].mean()) ** 2).sum()
                            vals.append(1 - (e ** 2).sum() / sst)
                    m, s = stat(vals)
                    cells.append(md(m, s))
                    raw.append((m, s))
                W(f"| {lab} | " + " | ".join(cells) + " |")
                rows.append((lab, raw))
            W("")
            body = ["\\begin{tabular}{lcccc}", "\\toprule",
                    "Model & " + " & ".join(
                        f"{n} ({UNITS[n]})" if met != "R2" else n
                        for n in NAMES) + " \\\\", "\\midrule"]
            best = [(min if met != "R2" else max)(r[1][c][0] for r in rows)
                    for c in range(4)]
            for lab, raw in rows:
                cs = []
                for c in range(4):
                    t = tex(*raw[c])
                    if abs(raw[c][0] - best[c]) < 1e-12:
                        t = "\\textbf{" + t + "}"
                    cs.append(t)
                nm = "\\textbf{" + lab + "}" if lab == LABEL else lab
                body.append(f"{nm} & " + " & ".join(cs) + " \\\\")
            body += ["\\bottomrule", "\\end{tabular}"]
            write_tex(os.path.join(ROOT, "paper", "tables",
                                   f"per_target_{met.lower()}_{ds}.tex"),
                      "\n".join(body),
                      f"Per-target {met} on {ds}, mean $\\pm$ s.d. over 5 seeds. "
                      f"Best per column in bold.",
                      f"tab:pt_{met.lower()}_{ds}")

        # ---------- per horizon ----------
        for met in ("MAE", "RMSE", "R2"):
            W(f"### Per horizon — {met} (all 4 targets averaged)\n")
            W("| Model | " + " | ".join(f"h={h}" for h in HZ) + " |")
            W("|---|" + "---|" * len(HZ))
            rows = []
            for model, abl in vs:
                lab = LABEL if (model, abl) == OURS else model
                ps = seeds_of(model, abl, ds)
                cells, raw = [], []
                for h in HZ:
                    vals = []
                    for p in ps:
                        e = p[:, h - 1, :] - true[:, h - 1, :]
                        if met == "MAE":
                            vals.append(np.abs(e).mean())
                        elif met == "RMSE":
                            vals.append(np.sqrt((e ** 2).mean()))
                        else:
                            t = true[:, h - 1, :]
                            sst = ((t - t.mean(axis=0, keepdims=True)) ** 2).sum(0)
                            vals.append(np.mean(1 - (e ** 2).sum(0) / sst))
                    m, s = stat(vals)
                    cells.append(md(m, s))
                    raw.append((m, s))
                W(f"| {lab} | " + " | ".join(cells) + " |")
                rows.append((lab, raw))
            W("")
            body = ["\\begin{tabular}{lcccc}", "\\toprule",
                    "Model & " + " & ".join(f"$h$={h}" for h in HZ) + " \\\\",
                    "\\midrule"]
            best = [(min if met != "R2" else max)(r[1][i][0] for r in rows)
                    for i in range(len(HZ))]
            for lab, raw in rows:
                cs = []
                for i in range(len(HZ)):
                    t = tex(*raw[i])
                    if abs(raw[i][0] - best[i]) < 1e-12:
                        t = "\\textbf{" + t + "}"
                    cs.append(t)
                nm = "\\textbf{" + lab + "}" if lab == LABEL else lab
                body.append(f"{nm} & " + " & ".join(cs) + " \\\\")
            body += ["\\bottomrule", "\\end{tabular}"]
            write_tex(os.path.join(ROOT, "paper", "tables",
                                   f"per_horizon_{met.lower()}_{ds}.tex"),
                      "\n".join(body),
                      f"Per-horizon {met} on {ds} (all four targets averaged), "
                      f"mean $\\pm$ s.d. over 5 seeds. Best per column in bold.",
                      f"tab:ph_{met.lower()}_{ds}")

    open(os.path.join(ROOT, "analysis", "per_target_horizon.md"),
         "w").write("\n".join(L))
    print("-> analysis/per_target_horizon.md")


if __name__ == "__main__":
    main()
