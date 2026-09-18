"""Dump validation-split predictions for every existing checkpoint.

Inference only. Nothing is trained, no saved artefact is overwritten, and
src/ is not modified -- `build_splits` and `evaluate` are imported from
src/train.py exactly as the training runs used them, so the scaler, the
split and the un-scaling are bit-identical to what produced
predictions/*_pred.npy.

    python analysis/dump_val_preds.py                 # val, all checkpoints
    python analysis/dump_val_preds.py --split test    # reproduction check

Outputs -> predictions_val/{tag}_valpred.npy  (N, 24, 4), float32
        -> predictions_val/{dataset}_val_true.npy
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data.dataset import build_splits                      # noqa: E402
from train import ABLATIONS, evaluate, set_seed            # noqa: E402
from xm_models.xai_meteoformer import XAIMeteoFormer       # noqa: E402
from baselines.tslib_adapter import build_baseline         # noqa: E402

# training defaults from src/train.py -- the runs used all of them unchanged
DEFAULTS = dict(seq_len=96, pred_len=24, patch_len=16, stride=8, d_model=256,
                n_heads=8, n_layers=2, dropout=0.2)


def parse_tag(fname):
    """{model}_{dataset}_{ablation}_s{seed}.pt -> parts, by matching known
    names rather than splitting on '_' (datasets and ablations contain it)."""
    from baselines.tslib_adapter import AVAILABLE
    stem = fname[:-3]
    seed = int(stem.rsplit("_s", 1)[1])
    rest = stem.rsplit("_s", 1)[0]
    for model in sorted(AVAILABLE, key=len, reverse=True):
        if not rest.startswith(model + "_"):
            continue
        tail = rest[len(model) + 1:]
        for abl in sorted(ABLATIONS, key=len, reverse=True):
            if tail.endswith("_" + abl):
                return model, tail[: -len(abl) - 1], abl, seed
    raise ValueError(f"cannot parse checkpoint name: {fname}")


def build(model_name, dataset, ablation, train_ds, device, tslib_path):
    args = SimpleNamespace(**DEFAULTS)
    if model_name == "XAI-MeteoFormer":
        model = XAIMeteoFormer(
            n_channels=train_ds.n_channels, target_idx=train_ds.target_idx,
            seq_len=args.seq_len, pred_len=args.pred_len,
            patch_len=args.patch_len, stride=args.stride,
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            dropout=args.dropout, **ABLATIONS[ablation],
        )
    else:
        # Historical configuration of the existing checkpoints: LSTM was
        # trained with our RevIN, every other baseline without any per-window
        # normalization of its own. Reproduced exactly so the weights load.
        historical = "on" if model_name in ("LSTM", "GRU") else "off"
        model = build_baseline(model_name, args, train_ds.n_channels,
                               train_ds.target_idx, train_ds.target_names,
                               tslib_path=tslib_path,
                               norm_variant=historical)
    return model.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--ckpt_dir", default=os.path.join(ROOT, "checkpoints"))
    p.add_argument("--out_dir", default=os.path.join(ROOT, "predictions_val"))
    p.add_argument("--processed_dir", default=os.path.join(ROOT, "data/processed"))
    p.add_argument("--tslib_path", default=os.path.join(ROOT, "Time-Series-Library"))
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                   else ("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--only", default=None, help="substring filter on the tag")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device(a.device)
    ckpts = sorted(f for f in os.listdir(a.ckpt_dir) if f.endswith(".pt"))
    if a.only:
        ckpts = [c for c in ckpts if a.only in c]
    print(f"{len(ckpts)} checkpoints -> {a.split} split on {device}")

    splits_cache = {}
    manifest = []
    for i, f in enumerate(ckpts, 1):
        model_name, dataset, ablation, seed = parse_tag(f)
        tag = f[:-3]
        suffix = "valpred" if a.split == "val" else "testpred"
        out = os.path.join(a.out_dir, f"{tag}_{suffix}.npy")
        if os.path.exists(out):
            print(f"[{i}/{len(ckpts)}] skip (exists) {tag}")
            continue

        if dataset not in splits_cache:
            # seed only feeds the missing-rate RNG, which is off; the split
            # and the scaler are seed-independent (see src/data/dataset.py)
            splits_cache[dataset] = build_splits(
                a.processed_dir, dataset, seq_len=DEFAULTS["seq_len"],
                pred_len=DEFAULTS["pred_len"], seed=0)
        train_ds, val_ds, test_ds = splits_cache[dataset]
        ds = val_ds if a.split == "val" else test_ds

        set_seed(seed)
        model = build(model_name, dataset, ablation, train_ds, device,
                      a.tslib_path)
        sd = torch.load(os.path.join(a.ckpt_dir, f), map_location=device)
        model.load_state_dict(sd)

        loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                            num_workers=0, drop_last=False)
        metrics, pred, true = evaluate(model, loader, device, train_ds.mu,
                                       train_ds.sigma, train_ds.target_idx,
                                       train_ds.target_names)
        np.save(out, pred.astype(np.float32))
        tf = os.path.join(a.out_dir, f"{dataset}_{a.split}_true.npy")
        if not os.path.exists(tf):
            np.save(tf, true.astype(np.float32))
        manifest.append({"tag": tag, "model": model_name, "dataset": dataset,
                         "ablation": ablation, "seed": seed,
                         "split": a.split, "n": int(pred.shape[0]),
                         "MAE": metrics["MAE"], "RMSE": metrics["RMSE"],
                         "R2": metrics["R2"]})
        print(f"[{i}/{len(ckpts)}] {tag:52s} n={pred.shape[0]:6d} "
              f"MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f}")
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    mf = os.path.join(a.out_dir, f"manifest_{a.split}.json")
    old = json.load(open(mf)) if os.path.exists(mf) else []
    json.dump(old + manifest, open(mf, "w"), indent=1)
    print(f"wrote {len(manifest)} arrays; manifest -> {mf}")


if __name__ == "__main__":
    main()
