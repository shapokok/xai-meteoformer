"""Error decomposition for the RMSE gap on Jena (diagnostic only).

Reads test-set predictions that were dumped at training time; trains
nothing, tunes nothing, writes nothing outside analysis/.

Variants compared: XAI-MeteoFormer full (RevIN on), XAI-MeteoFormer
no_revin, Crossformer full. Five seeds each, mean +- sd over seeds.

The amplitude axis is the excursion of the target inside the forecast
window, max_h |y_true[h] - y_last_input|, computed from ground truth
only -- it does not depend on any model.
"""

import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Defaults reproduce analysis/error_decomposition.md exactly. They are module
# level because the report text quotes them; --dataset / --variants override
# them for a new configuration, e.g. checkpoints trained with an MSE loss
# (tag suffix "_mse") or with the block augmentation ("_augblk0.5").
#
# NOTE: this script runs no model. It reads predictions/*.npy, so it needs no
# GPU and can run while something else is using the accelerator.
DS = "jena"
SEEDS = range(5)
VARIANTS = [
    ("XAI-MeteoFormer", "no_revin", "", "Ours (no_revin)"),
    ("XAI-MeteoFormer", "full", "", "Ours (RevIN)"),
    ("Crossformer", "full", "", "Crossformer"),
]
HORIZONS = [1, 6, 12, 24]
L, H = 96, 24


def load():
    meta = json.load(open(f"{ROOT}/data/processed/{DS}_meta.json"))
    names = meta["target_names"]
    X = np.load(f"{ROOT}/data/processed/{DS}_X.npy")
    T = X.shape[0]
    raw = X[int(T * 0.7) + int(T * 0.1):]
    true = np.load(f"{ROOT}/predictions/{DS}_true.npy").astype(np.float64)
    N = true.shape[0]
    # last observed value of each target, per window
    last = np.stack([raw[i + L - 1, meta["target_idx"]] for i in range(N)])
    return names, true, last


def preds(model, abl, sfx=""):
    """per-seed test predictions; sfx is the tag suffix written by train.py"""
    out = []
    for s in SEEDS:
        f = f"{ROOT}/predictions/{model}_{DS}_{abl}{sfx}_s{s}_pred.npy"
        if not os.path.exists(f):
            raise FileNotFoundError(f)
        out.append(np.load(f).astype(np.float64))
    return out


def ms(v):
    """mean +- sd over seeds"""
    return float(np.mean(v)), float(np.std(v, ddof=1))


def fmt(m, s):
    return f"{m:.3f}±{s:.3f}"


def main():
    names, true, last = load()
    P = {lab: preds(m, a, sfx) for m, a, sfx, lab in VARIANTS}
    labels = [lab for *_, lab in VARIANTS]
    out = []
    W = out.append

    W("# Error decomposition on Jena: where the RMSE gap lives\n")
    W("Diagnostic only. Nothing retrained, nothing tuned; the test set is read, "
      "never fitted. Numbers are mean±sd over 5 seeds, in physical units "
      "(T °C, RH %, P mbar, WS m/s). Aggregate rows average the four targets, "
      "which is how the paper's headline MAE/RMSE are defined, so they are "
      "dominated by whichever channel has the largest scale.\n")

    # ---- 0. overall ----
    W("## 0. Overall (reproduced from the dumped arrays)\n")
    W("| Variant | MAE | RMSE |")
    W("|---|---|---|")
    for lab in labels:
        mae = [np.abs(p - true).mean() for p in P[lab]]
        rmse = [np.sqrt(((p - true) ** 2).mean()) for p in P[lab]]
        W(f"| {lab} | {fmt(*ms(mae))} | {fmt(*ms(rmse))} |")
    W("")

    # ---- 1. per channel ----
    W("## 1. Per target channel\n")
    W("| Variant | metric | " + " | ".join(names) + " |")
    W("|---|---|" + "---|" * len(names))
    for lab in labels:
        for met in ("MAE", "RMSE"):
            cells = []
            for c in range(len(names)):
                e = [p[:, :, c] - true[:, :, c] for p in P[lab]]
                v = ([np.abs(x).mean() for x in e] if met == "MAE"
                     else [np.sqrt((x ** 2).mean()) for x in e])
                cells.append(fmt(*ms(v)))
            W(f"| {lab} | {met} | " + " | ".join(cells) + " |")
    W("")

    # ---- 2. per horizon ----
    W("## 2. Per horizon (all 4 targets averaged)\n")
    W("| Variant | metric | " + " | ".join(f"h={h}" for h in HORIZONS) + " |")
    W("|---|---|" + "---|" * len(HORIZONS))
    for lab in labels:
        for met in ("MAE", "RMSE"):
            cells = []
            for h in HORIZONS:
                e = [p[:, h - 1, :] - true[:, h - 1, :] for p in P[lab]]
                v = ([np.abs(x).mean() for x in e] if met == "MAE"
                     else [np.sqrt((x ** 2).mean()) for x in e])
                cells.append(fmt(*ms(v)))
            W(f"| {lab} | {met} | " + " | ".join(cells) + " |")
    W("")

    # ---- 2b. per horizon, T only ----
    W("### 2b. Per horizon, temperature only\n")
    W("| Variant | metric | " + " | ".join(f"h={h}" for h in HORIZONS) + " |")
    W("|---|---|" + "---|" * len(HORIZONS))
    for lab in labels:
        for met in ("MAE", "RMSE"):
            cells = []
            for h in HORIZONS:
                e = [p[:, h - 1, 0] - true[:, h - 1, 0] for p in P[lab]]
                v = ([np.abs(x).mean() for x in e] if met == "MAE"
                     else [np.sqrt((x ** 2).mean()) for x in e])
                cells.append(fmt(*ms(v)))
            W(f"| {lab} | {met} | " + " | ".join(cells) + " |")
    W("")

    # ---- 3. amplitude deciles ----
    # amp[n, c] = max_h |y_true[n,h,c] - y_last_input[n,c]|
    amp = np.abs(true - last[:, None, :]).max(axis=1)              # (N, C)
    # decile id per channel, from ground truth only
    dec = np.empty_like(amp, dtype=int)
    edges = {}
    for c in range(amp.shape[1]):
        q = np.quantile(amp[:, c], np.linspace(0, 1, 11))
        q[0], q[-1] = -np.inf, np.inf
        dec[:, c] = np.clip(np.searchsorted(q, amp[:, c], side="right") - 1, 0, 9)
        edges[c] = np.quantile(amp[:, c], np.linspace(0, 1, 11))

    W("## 3. Amplitude deciles\n")
    W("Each window gets, per channel, the excursion "
      "`max_h |y_true[h] - y_last_input|`; windows are ranked into deciles "
      "**within each channel**, so D1 = flattest 10 % of windows for that "
      "channel and D10 = the most volatile 10 %. Errors are then pooled over "
      "the 4 channels within a decile.\n")

    W("### 3a. Decile definition (amplitude range, ground truth only)\n")
    W("| decile | " + " | ".join(names) + " |")
    W("|---|" + "---|" * len(names))
    for d in range(10):
        cells = [f"{edges[c][d]:.2f}–{edges[c][d+1]:.2f}" for c in range(len(names))]
        W(f"| D{d+1} | " + " | ".join(cells) + " |")
    W("")

    def by_decile(p):
        """-> (10,) MAE and RMSE pooled over channels within each decile"""
        e = p - true                                               # (N,H,C)
        mae = np.empty(10)
        rmse = np.empty(10)
        for d in range(10):
            acc_abs, acc_sq, n = 0.0, 0.0, 0
            for c in range(e.shape[2]):
                sel = dec[:, c] == d
                ec = e[sel, :, c]
                acc_abs += np.abs(ec).sum()
                acc_sq += (ec ** 2).sum()
                n += ec.size
            mae[d] = acc_abs / n
            rmse[d] = np.sqrt(acc_sq / n)
        return mae, rmse

    tab = {lab: [by_decile(p) for p in P[lab]] for lab in labels}

    for met, idx in (("MAE", 0), ("RMSE", 1)):
        W(f"### 3b. {met} by amplitude decile (all targets pooled)\n")
        W("| Variant | " + " | ".join(f"D{d+1}" for d in range(10)) + " |")
        W("|---|" + "---|" * 10)
        for lab in labels:
            v = np.stack([t[idx] for t in tab[lab]])                # (seeds,10)
            W(f"| {lab} | " + " | ".join(
                fmt(v[:, d].mean(), v[:, d].std(ddof=1)) for d in range(10)) + " |")
        # deltas
        a = np.stack([t[idx] for t in tab[labels[0]]])
        b = np.stack([t[idx] for t in tab[labels[1]]])
        cf = np.stack([t[idx] for t in tab[labels[2]]])
        W(f"| **{labels[0]} − {labels[1]}** | " + " | ".join(
            f"{(a-b)[:, d].mean():+.3f}" for d in range(10)) + " |")
        W(f"| **{labels[0]} − {labels[2]}** | " + " | ".join(
            f"{(a-cf)[:, d].mean():+.3f}" for d in range(10)) + " |")
        W("")

    # ---- 3c. contribution to total squared error ----
    W("### 3c. Share of total squared error contributed by each decile\n")
    W("RMSE is a sum of squares, so a decile matters in proportion to the "
      "squared error it carries, not to its width. Share = "
      "(sum of squared errors in the decile) / (total), %.\n")
    W("| Variant | " + " | ".join(f"D{d+1}" for d in range(10)) + " |")
    W("|---|" + "---|" * 10)
    for lab in labels:
        sh = []
        for p in P[lab]:
            e = (p - true) ** 2
            s = np.array([sum(e[dec[:, c] == d, :, c].sum()
                              for c in range(e.shape[2])) for d in range(10)])
            sh.append(100 * s / s.sum())
        sh = np.stack(sh)
        W(f"| {lab} | " + " | ".join(f"{sh[:, d].mean():.1f}%" for d in range(10)) + " |")
    W("")

    # ---- 4. T-only deciles (the channel that drives the aggregate) ----
    W("### 3d. Temperature only, by its own amplitude decile\n")
    W("| Variant | metric | " + " | ".join(f"D{d+1}" for d in range(10)) + " |")
    W("|---|---|" + "---|" * 10)
    for lab in labels:
        for met in ("MAE", "RMSE"):
            per_seed = []
            for p in P[lab]:
                e = p[:, :, 0] - true[:, :, 0]
                row = [(np.abs(e[dec[:, 0] == d]).mean() if met == "MAE"
                        else np.sqrt((e[dec[:, 0] == d] ** 2).mean()))
                       for d in range(10)]
                per_seed.append(row)
            v = np.array(per_seed)
            W(f"| {lab} | {met} | " + " | ".join(
                fmt(v[:, d].mean(), v[:, d].std(ddof=1)) for d in range(10)) + " |")
    W("")

    # ---- 5. tail of the error distribution ----
    W("## 4. Tail of the error distribution\n")
    W("Absolute-error quantiles over all (window, horizon, channel) triples, "
      "and the share of total squared error carried by the worst 1 % of them.\n")
    W("| Variant | p50 | p90 | p99 | p99.9 | max | SE share of worst 1% |")
    W("|---|---|---|---|---|---|---|")
    for lab in labels:
        qs = {q: [] for q in (50, 90, 99, 99.9)}
        mx, share = [], []
        for p in P[lab]:
            a = np.abs(p - true).ravel()
            for q in qs:
                qs[q].append(np.percentile(a, q))
            mx.append(a.max())
            thr = np.percentile(a, 99)
            share.append(100 * (a[a >= thr] ** 2).sum() / (a ** 2).sum())
        W(f"| {lab} | " + " | ".join(fmt(*ms(qs[q])) for q in (50, 90, 99, 99.9))
          + f" | {fmt(*ms(mx))} | {fmt(*ms(share))} |")
    W("")


    # ---- 5. channel MSE budget ----
    W("## 5. Where the aggregate gap actually comes from: the channel budget\n")
    W("The headline MAE/RMSE are unweighted means over the four targets in "
      "physical units. RH has ~10x the error scale of T, so the aggregate is "
      "an RH statistic wearing a coat. Aggregate MSE = mean of the four "
      "channel MSEs, so each channel moves it by (its MSE gap)/4.\n")
    W("| Variant | MSE T | MSE RH | MSE P | MSE WS | aggregate MSE | aggregate RMSE |")
    W("|---|---|---|---|---|---|---|")
    mses = {}
    for lab in labels:
        m = np.array([[((p[:, :, c] - true[:, :, c]) ** 2).mean() for c in range(4)]
                      for p in P[lab]]).mean(0)
        mses[lab] = m
        W(f"| {lab} | " + " | ".join(f"{m[c]:.3f}" for c in range(4))
          + f" | {m.mean():.3f} | {np.sqrt(m.mean()):.3f} |")
    d = mses[labels[0]] - mses[labels[2]]
    W(f"| **{labels[0]} − {labels[2]}** | " + " | ".join(f"{d[c]:+.3f}" for c in range(4))
      + f" | {d.mean():+.3f} | — |")
    W("| **contribution to aggregate MSE gap** | "
      + " | ".join(f"{d[c]/4:+.3f}" for c in range(4))
      + f" | {d.mean():+.3f} | — |")
    W("")

    # ---- 6. RH deciles ----
    W("### 5b. RH only, by RH amplitude decile (RMSE)\n")
    W("| Variant | " + " | ".join(f"D{d+1}" for d in range(10)) + " |")
    W("|---|" + "---|" * 10)
    dc = dec[:, 1]
    rh = {}
    for lab in labels:
        v = np.array([[np.sqrt(((p[dc == d][:, :, 1] - true[dc == d][:, :, 1]) ** 2).mean())
                       for d in range(10)] for p in P[lab]])
        rh[lab] = v
        W(f"| {lab} | " + " | ".join(fmt(v[:, d].mean(), v[:, d].std(ddof=1))
                                     for d in range(10)) + " |")
    g = rh[labels[0]] - rh[labels[2]]
    W(f"| **{labels[0]} − {labels[2]}** | " + " | ".join(f"{g[:, d].mean():+.3f}"
                                                    for d in range(10)) + " |")
    W("")

    # ---- 7. Huber regime ----
    W("## 6. Is the loss the culprit? (Huber operating regime)\n")
    meta = json.load(open(f"{ROOT}/data/processed/{DS}_meta.json"))
    Xf = np.load(f"{ROOT}/data/processed/{DS}_X.npy")
    sg = Xf[:int(Xf.shape[0] * 0.7)].std(0)[meta["target_idx"]]
    W("Training uses `HuberLoss(delta=1.0)` on globally standardised targets "
      f"(train sd: " + ", ".join(f"{n}={sg[i]:.2f}" for i, n in enumerate(names))
      + "). Fraction of test residuals past delta, i.e. the share the loss "
        "treats as L1 rather than L2:\n")
    W("| Variant | overall | " + " | ".join(names) + " |")
    W("|---|---|" + "---|" * len(names))
    for lab in labels:
        s_ = sg.reshape(1, 1, -1)
        o = np.mean([(np.abs((p - true) / s_) > 1.0).mean() for p in P[lab]])
        pc = np.array([[(np.abs((p[:, :, c] - true[:, :, c]) / sg[c]) > 1.0).mean()
                        for c in range(4)] for p in P[lab]]).mean(0)
        W(f"| {lab} | {o:.3f} | " + " | ".join(f"{pc[c]:.3f}" for c in range(4)) + " |")
    W("\nAbout 93 % of residuals sit in the quadratic region, so the objective "
      "is effectively MSE for every model here. The loss does not explain an "
      "MAE-good / RMSE-bad profile.\n")

    # ---- 8. dispersion ----
    W("## 7. What does explain it: RH dispersion\n")
    W("Ratio of predicted sd to ground-truth sd per channel, and the scalar "
      "`a` that would minimise RH MSE under `pred -> mean + a*(pred - mean)`. "
      "`a < 1` means the forecast is over-dispersed for a squared loss.\n")
    W("| Variant | sd ratio T | sd ratio RH | sd ratio P | sd ratio WS | oracle a (RH) |")
    W("|---|---|---|---|---|---|")
    tsd = np.array([true[:, :, c].std() for c in range(4)])
    for lab in labels:
        r = np.array([[p[:, :, c].std() for c in range(4)] for p in P[lab]]).mean(0) / tsd
        aa = []
        for p in P[lab]:
            x, y = p[:, :, 1], true[:, :, 1]
            m = x.mean()
            aa.append(((y - m) * (x - m)).sum() / ((x - m) ** 2).sum())
        W(f"| {lab} | " + " | ".join(f"{r[c]:.3f}" for c in range(4))
          + f" | {np.mean(aa):.3f} |")
    W("")
    W("Effect of applying that RH rescale (oracle, fitted **on test** — a "
      "measurement of how much of the gap is dispersion, **not** a proposed "
      "method; an honest version must fit `a` on validation):\n")
    W("| Variant | aggregate RMSE as-is | with RH rescaled |")
    W("|---|---|---|")
    for lab in labels:
        base, resc = [], []
        for p in P[lab]:
            m4 = [((p[:, :, c] - true[:, :, c]) ** 2).mean() for c in range(4)]
            x, y = p[:, :, 1], true[:, :, 1]
            mm = x.mean()
            a = ((y - mm) * (x - mm)).sum() / ((x - mm) ** 2).sum()
            m4b = list(m4); m4b[1] = (((mm + a * (x - mm)) - y) ** 2).mean()
            base.append(np.sqrt(np.mean(m4))); resc.append(np.sqrt(np.mean(m4b)))
        W(f"| {lab} | {fmt(*ms(base))} | {fmt(*ms(resc))} |")
    W("")
    os.makedirs(f"{ROOT}/analysis", exist_ok=True)
    open(OUT, "w").write("\n".join(out))
    print("\n".join(out))


def cli():
    global DS, SEEDS, VARIANTS, OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DS)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--variants", nargs="*", default=None,
                    help="model|ablation|suffix|label , first must be the "
                         "variant under test, third the comparison baseline")
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis", "_tables.md"))
    a = ap.parse_args()
    DS, SEEDS, OUT = a.dataset, a.seeds, a.out
    if a.variants:
        VARIANTS = [tuple(v.split("|")) for v in a.variants]
        assert all(len(v) == 4 for v in VARIANTS), "need model|ablation|suffix|label"
    main()


OUT = os.path.join(ROOT, "analysis", "_tables.md")

if __name__ == "__main__":
    cli()
