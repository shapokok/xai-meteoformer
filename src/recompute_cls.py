"""Fair event-classification metrics, recomputed from saved predictions.

Why this exists. Baselines have no event head, so their frost score is
derived from the predicted temperature. That score carries the model's
full regression supervision, while a dedicated binary head sees only a
0/1 label — so comparing our head against their thresholded regression
compares two different amounts of information, not two models.

This script scores EVERY model the same way, straight from the saved
`*_pred.npy`:

    ranking score for ROC / PR : -T_pred
    hard decision              : T_pred <= 0 degC

The decision threshold is the physical freezing point, so nothing is
tuned and nothing leaks from the test set.

    python src/recompute_cls.py --out_dir outputs --dataset jena

Writes outputs/cls_metrics.csv — the source for Table 12 and Figures
16-17. The `CLS_*` columns in results.csv stay as they are: for
XAI-MeteoFormer they measure the dedicated head, which is a separate
question (does the head add anything over thresholding the regression?)
and an honest ablation row either way.
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="frost threshold in degC")
    args = ap.parse_args()

    pred_dir = os.path.join(args.out_dir, "predictions")
    true_path = os.path.join(pred_dir, f"{args.dataset}_true.npy")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"{true_path} missing — run training first")

    with open(os.path.join(args.processed_dir,
                           f"{args.dataset}_meta.json")) as f:
        meta = json.load(f)
    t_pos = meta["target_names"].index("T")

    true = np.load(true_path)                      # (N, H, n_targets)
    y = (true[:, :, t_pos] <= args.threshold).astype(int).ravel()
    print(f"{args.dataset}: {len(y)} (sample, step) pairs, "
          f"positive rate {y.mean():.3f}")

    rows = []
    pat = re.compile(rf"^(.+)_{re.escape(args.dataset)}_(.+)_s(\d+)_pred\.npy$")
    for fn in sorted(os.listdir(pred_dir)):
        m = pat.match(fn)
        if not m:
            continue
        model, ablation, seed = m.group(1), m.group(2), int(m.group(3))
        pred = np.load(os.path.join(pred_dir, fn))
        if pred.shape != true.shape:
            print(f"  skip {fn}: shape {pred.shape} != {true.shape}")
            continue

        t_pred = pred[:, :, t_pos].ravel()
        score = -t_pred                              # higher = more frost
        yhat = (t_pred <= args.threshold).astype(int)

        tn, fp, fn_, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        rows.append({
            "model": model, "dataset": args.dataset, "ablation": ablation,
            "seed": seed,
            "AUC": roc_auc_score(y, score),
            "AP": average_precision_score(y, score),
            "F1": f1_score(y, yhat, zero_division=0),
            "Precision": precision_score(y, yhat, zero_division=0),
            "Recall": recall_score(y, yhat, zero_division=0),
            "TN": tn, "FP": fp, "FN": fn_, "TP": tp,
            "pos_rate": float(y.mean()),
        })

    if not rows:
        raise RuntimeError(f"no *_pred.npy for '{args.dataset}' in {pred_dir}")

    df = pd.DataFrame(rows)
    out = os.path.join(args.out_dir, "cls_metrics.csv")
    df.to_csv(out, index=False)

    summary = (df.groupby("model")[["AUC", "AP", "F1", "Precision", "Recall"]]
                 .agg(["mean", "std"]).round(4).sort_values(("AP", "mean"),
                                                            ascending=False))
    print(summary.to_string())
    print(f"\nwrote {out} ({len(df)} runs)")


if __name__ == "__main__":
    main()
