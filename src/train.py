"""Training runner for XAI-MeteoFormer.

Built for Kaggle's 12-hour session cap: results are appended to a CSV
after every single run and any (model, dataset, seed) already present is
skipped. A session that dies at hour 12 loses one run, not the batch.

    python src/train.py --dataset jena --seed 0
    python src/train.py --dataset jena --seeds 0 1 2 3 4      # sweep seeds
    python src/train.py --dataset jena --ablation no_var_attn

Metric notes for the paper:
  * MAPE is reported ONLY for pressure and relative humidity. Temperature
    in Celsius crosses zero, so percentage error explodes; sMAPE is
    reported for every target instead.
  * Errors are broken down at h = 1, 6, 12, 24 from a single trained
    model — the horizon axis costs no extra training runs.
"""

import argparse
import itertools
import json
import os
import random
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import build_splits          # noqa: E402
from xm_models.xai_meteoformer import XAIMeteoFormer  # noqa: E402
from baselines.tslib_adapter import AVAILABLE, build_baseline  # noqa: E402

ABLATIONS = {
    "full": {},
    "no_revin": {"use_revin": False},
    "no_multiscale": {"use_multiscale": False},
    "no_var_attn": {"use_var_attn": False},
    "no_temp_attn": {"use_temp_attn": False},
    "no_fusion": {"use_fusion": False},
    "no_entropy": {},          # handled via --lambda_ent 0
    "no_cls": {},              # handled via --lambda_cls 0
}
REPORT_HORIZONS = [1, 6, 12, 24]

# These models run an FFT inside the forward pass. Under AMP the transform
# is done in half precision, and cuFFT then requires a power-of-two signal
# length — our seq_len is 96, so it raises. Costs a little speed, nothing
# else: the comparison stays fair because the maths is identical, only the
# arithmetic precision differs.
AMP_UNSAFE = {"Autoformer", "FEDformer", "TimesNet", "FiLM"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def regression_metrics(pred: np.ndarray, true: np.ndarray,
                       names: List[str]) -> Dict[str, float]:
    """pred/true: (n_samples, H, n_targets) in original units."""
    out: Dict[str, float] = {}
    err = pred - true
    out["MAE"] = float(np.abs(err).mean())
    out["RMSE"] = float(np.sqrt((err ** 2).mean()))
    out["MSE"] = float((err ** 2).mean())

    denom = np.abs(pred) + np.abs(true)
    out["SMAPE"] = float(np.mean(2.0 * np.abs(err) / np.clip(denom, 1e-6, None)) * 100)

    ss_res = ((err) ** 2).sum(axis=(0, 1))
    ss_tot = ((true - true.mean(axis=(0, 1), keepdims=True)) ** 2).sum(axis=(0, 1))
    out["R2"] = float(np.mean(1.0 - ss_res / np.clip(ss_tot, 1e-9, None)))

    for i, n in enumerate(names):
        e = err[:, :, i]
        out[f"MAE_{n}"] = float(np.abs(e).mean())
        out[f"RMSE_{n}"] = float(np.sqrt((e ** 2).mean()))
        # MAPE only where the target cannot sit near zero
        if n in ("P", "RH"):
            out[f"MAPE_{n}"] = float(
                np.mean(np.abs(e) / np.clip(np.abs(true[:, :, i]), 1e-6, None)) * 100
            )

    for h in REPORT_HORIZONS:
        if h <= pred.shape[1]:
            e = err[:, h - 1, :]
            out[f"MAE_h{h}"] = float(np.abs(e).mean())
            out[f"RMSE_h{h}"] = float(np.sqrt((e ** 2).mean()))
    return out


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score,
                                 average_precision_score)
    p = 1.0 / (1.0 + np.exp(-logits.ravel()))
    y = labels.ravel().astype(int)
    if y.min() == y.max():          # degenerate season, no positives
        return {"CLS_note": float("nan")}
    yhat = (p >= 0.5).astype(int)
    return {
        "CLS_ACC": float(accuracy_score(y, yhat)),
        "CLS_P": float(precision_score(y, yhat, zero_division=0)),
        "CLS_R": float(recall_score(y, yhat, zero_division=0)),
        "CLS_F1": float(f1_score(y, yhat, zero_division=0)),
        "CLS_AUC": float(roc_auc_score(y, p)),
        "CLS_AP": float(average_precision_score(y, p)),
        "CLS_POSRATE": float(y.mean()),
    }


def already_done(csv_path: str, key: Dict[str, object]) -> bool:
    if not os.path.exists(csv_path):
        return False
    df = pd.read_csv(csv_path)
    m = np.ones(len(df), dtype=bool)
    for k, v in key.items():
        if k not in df.columns:
            return False
        m &= (df[k].astype(str) == str(v)).values
    return bool(m.any())


def append_row(csv_path: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)


def maybe_limit(loader, lim):
    """Truncate a DataLoader for smoke runs. Debug on CPU, never on quota."""
    return itertools.islice(loader, lim) if lim else loader


@torch.no_grad()
def evaluate(model, loader, device, mu, sigma, target_idx, names, limit=None):
    model.eval()
    preds, trues, logits, labels = [], [], [], []
    for b in maybe_limit(loader, limit):
        x = b["x"].to(device, non_blocking=True)
        out = model(x)
        preds.append(out["y_pred"].float().cpu().numpy())
        logits.append(out["logits"].float().cpu().numpy())
        trues.append(b["y_raw"].numpy())
        labels.append(b["cls"].numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)

    # model outputs live in globally-scaled units -> back to physical units
    s = sigma[target_idx].reshape(1, 1, -1)
    m = mu[target_idx].reshape(1, 1, -1)
    pred = pred * s + m

    metrics = regression_metrics(pred, true, names)
    metrics.update(classification_metrics(np.concatenate(logits),
                                          np.concatenate(labels)))
    return metrics, pred, true


def run_one(args, seed: int) -> None:
    lim = args.smoke_batches if args.smoke else None
    # Smoke artefacts go to their own folder. Otherwise the truncated
    # ground-truth file written here would be picked up by the real run,
    # which only writes it when absent.
    out_dir = os.path.join(args.out_dir, "smoke") if args.smoke else args.out_dir
    results_csv = os.path.join(out_dir, "smoke.csv") if args.smoke \
        else args.results_csv
    if args.smoke:
        args.epochs, args.patience = 1, 1
        print(f"SMOKE MODE: 1 epoch, {lim} batches, artefacts -> {out_dir}/")

    # lambda_ent belongs in the key: without it a sweep over the entropy
    # weight would collide with the default run and every point after the
    # first would be skipped as "already done".
    lam_key = (0.0 if (args.model_name != "XAI-MeteoFormer"
                       or args.ablation == "no_entropy")
               else args.lambda_ent)
    key = {"model": args.model_name, "dataset": args.dataset,
           "ablation": args.ablation, "seed": seed,
           "missing_rate": args.missing_rate, "lambda_ent": lam_key}
    if already_done(results_csv, key) and not args.force:
        print(f"skip (already in results): {key}")
        return

    set_seed(seed)
    device = torch.device(args.device)

    train_ds, val_ds, test_ds = build_splits(
        args.processed_dir, args.dataset,
        seq_len=args.seq_len, pred_len=args.pred_len, seed=seed,
    )
    test_ds.missing_rate = args.missing_rate

    dl = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              pin_memory=True, drop_last=False)
    train_dl = DataLoader(train_ds, shuffle=True, **dl)
    val_dl = DataLoader(val_ds, shuffle=False, **dl)
    test_dl = DataLoader(test_ds, shuffle=False, **dl)

    if args.model_name == "XAI-MeteoFormer":
        kw = dict(ABLATIONS[args.ablation])
        lambda_ent = 0.0 if args.ablation == "no_entropy" else args.lambda_ent
        if args.ablation == "no_cls":
            # Does the auxiliary event loss actually help the regression?
            # If not, the head can be dropped from the model entirely.
            args.lambda_cls = 0.0
        model = XAIMeteoFormer(
            n_channels=train_ds.n_channels, target_idx=train_ds.target_idx,
            seq_len=args.seq_len, pred_len=args.pred_len,
            patch_len=args.patch_len, stride=args.stride,
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            dropout=args.dropout, **kw,
        ).to(device)
    else:
        # Baselines share the loop, the split and the loss. Only the
        # entropy term is dropped — they have no variable attention.
        lambda_ent = 0.0
        model = build_baseline(
            args.model_name, args, train_ds.n_channels,
            train_ds.target_idx, train_ds.target_names,
            tslib_path=args.tslib_path,
        ).to(device)

    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_dl),
        pct_start=0.3,
    )
    reg_loss = nn.HuberLoss(delta=1.0)
    cls_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [max(1.0, (1 - train_ds.frost_rate) / max(train_ds.frost_rate, 1e-3))],
            device=device)
    )
    use_amp = (args.amp and not args.no_amp and device.type == "cuda"
               and args.model_name not in AMP_UNSAFE)
    if args.model_name in AMP_UNSAFE and device.type == "cuda":
        print(f"[{args.model_name}] AMP off (FFT needs fp32 at seq_len="
              f"{args.seq_len})")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    # keep the default tag unchanged so existing checkpoints stay valid
    suffix = "" if lam_key == 0.01 or lam_key == 0.0 else f"_le{lam_key}"
    tag = f"{args.model_name}_{args.dataset}_{args.ablation}{suffix}_s{seed}"
    ckpt_path = os.path.join(ckpt_dir, f"{tag}.pt")

    best, patience, t0 = float("inf"), 0, time.time()
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        n_seen = 0
        for b in maybe_limit(train_dl, lim):
            x = b["x"].to(device, non_blocking=True)
            y = b["y"].to(device, non_blocking=True)
            c = b["cls"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x)
                loss = (reg_loss(out["y_pred"], y)
                        + args.lambda_cls * cls_loss(out["logits"], c)
                        + lambda_ent * out["var_entropy"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * x.size(0)
            n_seen += x.size(0)

        model.eval()
        vtot, v_seen = 0.0, 0
        with torch.no_grad():
            for b in maybe_limit(val_dl, lim):
                x = b["x"].to(device)
                y = b["y"].to(device)
                vtot += reg_loss(model(x)["y_pred"], y).item() * x.size(0)
                v_seen += x.size(0)
        vloss = vtot / max(v_seen, 1)
        print(f"[{tag}] ep {ep+1}/{args.epochs} "
              f"train {tot/max(n_seen,1):.4f} val {vloss:.4f}")

        if vloss < best - 1e-5:
            best, patience = vloss, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[{tag}] early stop at epoch {ep+1}")
                break

    train_time = time.time() - t0
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    t1 = time.time()
    metrics, pred, true = evaluate(model, test_dl, device,
                                   train_ds.mu, train_ds.sigma,
                                   train_ds.target_idx, train_ds.target_names,
                                   limit=lim)
    infer_time = time.time() - t1

    pred_dir = os.path.join(out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    np.save(os.path.join(pred_dir, f"{tag}_pred.npy"), pred.astype(np.float32))
    if not os.path.exists(os.path.join(pred_dir, f"{args.dataset}_true.npy")):
        np.save(os.path.join(pred_dir, f"{args.dataset}_true.npy"),
                true.astype(np.float32))

    row = {**key, "params": n_par, "train_time_s": round(train_time, 1),
           "infer_time_s": round(infer_time, 2), "best_val": best,
           "epochs_run": ep + 1, "seq_len": args.seq_len,
           "pred_len": args.pred_len, "d_model": args.d_model,
           "n_layers": args.n_layers, "lr": args.lr,
           **metrics}
    append_row(results_csv, row)
    print(json.dumps({k: v for k, v in row.items()
                      if k in ("MAE", "RMSE", "R2", "CLS_AUC", "CLS_F1")}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--results_csv", default="outputs/results.csv")
    p.add_argument("--model_name", default="XAI-MeteoFormer", choices=AVAILABLE)
    p.add_argument("--models", nargs="*", default=None,
                   help="run several models in one go, e.g. --models DLinear PatchTST")
    p.add_argument("--tslib_path", default=None,
                   help="path to Time-Series-Library (or set $TSLIB_PATH)")
    p.add_argument("--ablation", default="full", choices=list(ABLATIONS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=24)
    p.add_argument("--patch_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--lambda_cls", type=float, default=0.2)
    p.add_argument("--lambda_ent", type=float, default=0.01)
    p.add_argument("--missing_rate", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_true", help="force fp32")
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch on a handful of batches; for verifying the "
                        "pipeline on CPU before spending GPU quota")
    p.add_argument("--smoke_batches", type=int, default=20)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    models = args.models if args.models else [args.model_name]
    failed = []
    for m in models:
        args.model_name = m
        for s in (args.seeds if args.seeds else [args.seed]):
            try:
                run_one(args, s)
            except Exception as exc:
                # One broken baseline must not abort a 12-hour batch, but the
                # process must still exit non-zero so a caller (the Kaggle
                # notebook's preflight) can tell that nothing was produced.
                import traceback
                traceback.print_exc()
                print(f"!! {m} seed {s} FAILED: {type(exc).__name__}: {exc}")
                failed.append((m, s))
    if failed:
        print(f"FAILED runs: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
