"""Frost-event classification, scored identically for every model.

Reviewer 2 asked three things explicitly: how the regression baselines
were adapted to the event task, how the decision threshold was chosen,
and how class imbalance was handled. This script answers all three and
reports F1 next to AUC.

Adaptation. Only XAI-MeteoFormer has a dedicated event head. Every model
-- ours included -- is therefore scored here from its predicted
temperature, so the comparison is between equal amounts of information:

    ranking score (AUC / AP) : -T_pred
    hard decision            : T_pred <= tau

Threshold. Two variants are reported side by side:
  (a) tau = 0 degC, the physical freezing point. Nothing is fitted.
  (b) tau selected per (model, seed) by maximising F1 on the VALIDATION
      split, then frozen and applied to test. Never fitted on test.

Imbalance. Positives are rare. Nothing is resampled: AP and F1 are
reported alongside AUC because AUC is optimistic under imbalance, and
the positive rate is stated. In training, our event head used
BCEWithLogitsLoss(pos_weight=(1-r)/r) with r the frost rate of the
training split; baselines have no event loss at all.

Outputs -> analysis/frost_events.md, paper/tables/events_*.tex
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED, VAL = os.path.join(ROOT, "predictions"), os.path.join(ROOT, "predictions_val")
DATASETS = ["jena", "beijing_aotizhongxin"]
OURS, LABEL = ("XAI-MeteoFormer", "no_revin"), "MeteoFormer"
SEEDS = range(5)


def variants(ds):
    v = [OURS]
    for f in sorted(os.listdir(PRED)):
        if f.endswith("_pred.npy") and f"_{ds}_full_" in f:
            m = f.split(f"_{ds}_")[0]
            if m != OURS[0] and (m, "full") not in v:
                v.append((m, "full"))
    return v


def scores(y, t_pred, tau):
    yhat = (t_pred <= tau).astype(int)
    return {"AUC": roc_auc_score(y, -t_pred), "AP": average_precision_score(y, -t_pred),
            "F1": f1_score(y, yhat, zero_division=0),
            "Precision": precision_score(y, yhat, zero_division=0),
            "Recall": recall_score(y, yhat, zero_division=0)}


def best_tau(yv, tv):
    """tau maximising F1 on validation; searched over observed quantiles"""
    grid = np.unique(np.quantile(tv, np.linspace(0.0005, 0.20, 400)))
    grid = np.concatenate([grid, [0.0]])
    f1 = [f1_score(yv, (tv <= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1))]), float(np.max(f1))


def write_tex(path, body, caption, label):
    with open(path, "w") as f:
        f.write("\\begin{table}[H]\n\\caption{" + caption + "}\n")
        f.write("\\label{" + label + "}\n" + body + "\n\\end{table}\n")
    print("  wrote", os.path.relpath(path, ROOT))


def main():
    L = []
    W = L.append
    W("# Frost-event classification (Reviewer 2)\n")
    W("Produced by [analysis/frost_events.py](analysis/frost_events.py).\n")
    W("**Adaptation of the regression baselines.** No baseline has an event "
      "head. Every model, ours included, is scored from its predicted "
      "temperature: the ranking score for AUC/AP is `-T_pred`, the hard "
      "decision is `T_pred <= tau`. Scoring our dedicated head against the "
      "baselines' thresholded regression would compare two different amounts "
      "of supervision, not two models, so the head is evaluated separately in "
      "the ablation instead.\n")
    W("**Threshold.** Two variants, side by side. (a) `tau = 0 °C`, the "
      "physical freezing point — nothing is fitted, which is the cleanest "
      "possible claim. (b) `tau` maximising F1 on the **validation** split per "
      "(model, seed), then frozen and applied to test. The test set is never "
      "used to pick `tau`.\n")
    W("**Imbalance.** Nothing is resampled. The positive rate is stated below; "
      "AP and F1 are reported next to AUC because AUC is optimistic under "
      "imbalance. During training our event head used "
      "`BCEWithLogitsLoss(pos_weight=(1-r)/r)` with `r` the frost rate of the "
      "training split; the baselines have no event loss at all.\n")

    for ds in DATASETS:
        tf = os.path.join(PRED, f"{ds}_true.npy")
        vtf = os.path.join(VAL, f"{ds}_val_true.npy")
        if not os.path.exists(tf):
            continue
        meta = json.load(open(os.path.join(ROOT, "data/processed",
                                           f"{ds}_meta.json")))
        tp = meta["target_names"].index("T")
        true = np.load(tf).astype(np.float64)
        y = (true[:, :, tp] <= 0.0).astype(int).ravel()
        has_val = os.path.exists(vtf)
        if has_val:
            vtrue = np.load(vtf).astype(np.float64)
            yv = (vtrue[:, :, tp] <= 0.0).astype(int).ravel()

        title = "Jena" if ds == "jena" else "Beijing (Aotizhongxin)"
        W(f"\n---\n\n## {title}\n")
        W(f"Test positive rate: **{y.mean():.4f}** "
          f"({y.sum()} of {len(y)} (window, step) pairs).")
        if has_val:
            W(f"Validation positive rate: {yv.mean():.4f}.\n")
        else:
            W("_Validation predictions incomplete — variant (b) omitted._\n")

        rows = []
        for model, abl in variants(ds):
            lab = LABEL if (model, abl) == OURS else model
            a, b, taus = [], [], []
            for s in SEEDS:
                f = os.path.join(PRED, f"{model}_{ds}_{abl}_s{s}_pred.npy")
                if not os.path.exists(f):
                    continue
                t_pred = np.load(f).astype(np.float64)[:, :, tp].ravel()
                a.append(scores(y, t_pred, 0.0))
                vf = os.path.join(VAL, f"{model}_{ds}_{abl}_s{s}_valpred.npy")
                if has_val and os.path.exists(vf):
                    tv = np.load(vf).astype(np.float64)[:, :, tp].ravel()
                    tau, _ = best_tau(yv, tv)
                    taus.append(tau)
                    b.append(scores(y, t_pred, tau))
            if not a:
                continue
            rows.append({"model": lab, "n_a": len(a), "n_b": len(b),
                         "tau": np.mean(taus) if taus else np.nan,
                         "tau_sd": np.std(taus, ddof=1) if len(taus) > 1 else 0.0,
                         **{f"a_{k}": (np.mean([x[k] for x in a]),
                                       np.std([x[k] for x in a], ddof=1))
                            for k in a[0]},
                         **{f"b_{k}": (np.mean([x[k] for x in b]),
                                       np.std([x[k] for x in b], ddof=1))
                            for k in (b[0] if b else {})}})

        def cell(r, k):
            if k not in r or r[k] is None:
                return "—"
            m, s = r[k]
            return f"{m:.3f}±{s:.3f}"

        W("### (a) Physical threshold τ = 0 °C (nothing fitted)\n")
        W("| Model | AUC | AP | F1 | Precision | Recall |")
        W("|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: -r["a_AP"][0]):
            W(f"| {r['model']} | " + " | ".join(
                cell(r, f"a_{k}") for k in
                ("AUC", "AP", "F1", "Precision", "Recall")) + " |")
        W("")

        if any(r["n_b"] for r in rows):
            W("### (b) τ selected on validation to maximise F1\n")
            W("| Model | τ (°C) | AUC | AP | F1 | Precision | Recall |")
            W("|---|---|---|---|---|---|---|")
            for r in sorted(rows, key=lambda r: -r["a_AP"][0]):
                if not r["n_b"]:
                    W(f"| {r['model']} | — | — | — | — | — | — |")
                    continue
                W(f"| {r['model']} | {r['tau']:.2f}±{r['tau_sd']:.2f} | "
                  + " | ".join(cell(r, f"b_{k}") for k in
                               ("AUC", "AP", "F1", "Precision", "Recall")) + " |")
            W("")
            W("AUC and AP are threshold-free and therefore identical between "
              "(a) and (b); only F1/Precision/Recall move.\n")

        body = ["\\begin{tabular}{lccccc}", "\\toprule",
                "Model & AUC & AP & F1 & Precision & Recall \\\\", "\\midrule"]
        for r in sorted(rows, key=lambda r: -r["a_AP"][0]):
            nm = "\\textbf{" + r["model"] + "}" if r["model"] == LABEL else r["model"]
            body.append(f"{nm} & " + " & ".join(
                f"{r[f'a_{k}'][0]:.3f} $\\pm$ {r[f'a_{k}'][1]:.3f}"
                for k in ("AUC", "AP", "F1", "Precision", "Recall")) + " \\\\")
        body += ["\\bottomrule", "\\end{tabular}"]
        write_tex(os.path.join(ROOT, "paper", "tables", f"events_{ds}.tex"),
                  "\n".join(body),
                  f"Frost-event detection on {ds} (T $\\le$ 0\\,$^\\circ$C), "
                  f"mean $\\pm$ s.d. over 5 seeds. Every model is scored from "
                  f"its predicted temperature at the physical freezing point, "
                  f"so no threshold is fitted. Positive rate "
                  f"{y.mean():.4f}.", f"tab:events_{ds}")

    open(os.path.join(ROOT, "analysis", "frost_events.md"), "w").write("\n".join(L))
    print("-> analysis/frost_events.md")


if __name__ == "__main__":
    main()
