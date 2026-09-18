"""Block (contiguous) gaps, an operational failure mode (P0-5).

analysis/missing_robustness.md showed that per-cell gaps are almost free,
because they are repaired by linear interpolation along time and hourly
meteorological series are nearly linear over a few hours. That measures
the interpolator, not the model.

This is the harder, operationally realistic version: a whole BLOCK of
consecutive hours is lost across every channel, as when a station drops
out. Two placements with an identical missing fraction:

    A  block at the END of the input window   (t = 80..96, the most recent 16 h)
    B  block at the START of the window       (t = 0..16,  the oldest 16 h)

Both remove 16 of 96 timesteps = 16.7% of the input cells. Repair uses the
SAME rule as the per-cell experiment (linear interpolation along time over
the surviving indices, per channel) -- which for a terminal block degenerates
to holding the last valid value, exactly as an operational forward-fill would.

This doubles as an external check on analysis/occlusion_time.md: if the model
really only uses the last 16 h, A must hurt far more than B.

Inference only, on the existing checkpoints, on the same test subsample.

Output -> analysis/block_missing.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from data.dataset import build_splits                      # noqa: E402
from train import ABLATIONS, set_seed                      # noqa: E402
from xm_models.xai_meteoformer import XAIMeteoFormer       # noqa: E402
from baselines.tslib_adapter import build_baseline         # noqa: E402

DEFAULTS = dict(seq_len=96, pred_len=24, patch_len=16, stride=8, d_model=256,
                n_heads=8, n_layers=2, dropout=0.2)
NAMES = ["T", "RH", "P", "WS"]
SEQ, BLOCK = 96, 16
VARIANTS = {"none": None, "A_end": (SEQ - BLOCK, SEQ), "B_start": (0, BLOCK)}
# analysis/mps_unfold_fix.py lets the patching models run on MPS too
import mps_unfold_fix  # noqa: E402,F401
CPU_ONLY = set()


def mask_block(x, lo, hi):
    """x: (B, L, C) scaled. Blank t in [lo,hi) on every channel, then refill
    by linear interpolation along time -- the same rule dataset.py uses."""
    x = x.clone()
    keep = np.setdiff1d(np.arange(SEQ), np.arange(lo, hi))
    xk = x[:, keep, :].numpy()
    grid = np.arange(SEQ)
    out = np.empty(x.shape, dtype=np.float32)
    for b in range(x.shape[0]):
        for c in range(x.shape[2]):
            out[b, :, c] = np.interp(grid, keep, xk[b, :, c])
    return torch.from_numpy(out)


def build(model_name, ablation, train_ds, device, tslib):
    from types import SimpleNamespace
    args = SimpleNamespace(**DEFAULTS)
    if model_name == "XAI-MeteoFormer":
        m = XAIMeteoFormer(n_channels=train_ds.n_channels,
                           target_idx=train_ds.target_idx, **DEFAULTS,
                           **ABLATIONS[ablation])
    else:
        hist = "on" if model_name in ("LSTM", "GRU") else "off"
        m = build_baseline(model_name, args, train_ds.n_channels,
                           train_ds.target_idx, train_ds.target_names,
                           tslib_path=tslib, norm_variant=hist)
    return m.to(device)


@torch.no_grad()
def run(model, ds, dev, mu, sigma, tidx, span, stride, bs=128):
    view = Subset(ds, list(range(0, len(ds), stride))) if stride > 1 else ds
    P, T = [], []
    for b in DataLoader(view, batch_size=bs, shuffle=False, num_workers=0):
        x = b["x"] if span is None else mask_block(b["x"], *span)
        P.append(model(x.to(dev))["y_pred"].float().cpu().numpy())
        T.append(b["y_raw"].numpy())
    s = sigma[tidx].reshape(1, 1, -1)
    m = mu[tidx].reshape(1, 1, -1)
    return np.concatenate(P) * s + m, np.concatenate(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=["jena", "beijing_aotizhongxin"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--target_windows", type=int, default=2800)
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis",
                                                  "block_missing.csv"))
    a = ap.parse_args()
    tslib = os.path.join(ROOT, "Time-Series-Library")
    ck = os.path.join(ROOT, "checkpoints")

    rows = pd.read_csv(a.out).to_dict("records") if os.path.exists(a.out) else []
    done = {(r["dataset"], r["model"], r["seed"], r["variant"]) for r in rows}

    for ds in a.datasets:
        train_ds, _, test_ds = build_splits(
            os.path.join(ROOT, "data/processed"), ds, seq_len=SEQ,
            pred_len=24, seed=0)
        stride = max(1, round(len(test_ds) / a.target_windows))
        models = sorted({f.split(f"_{ds}_")[0] for f in os.listdir(ck)
                         if f.endswith(".pt") and f"_{ds}_" in f})
        for mn in models:
            abl = "no_revin" if mn == "XAI-MeteoFormer" else "full"
            dev = torch.device("cpu" if mn in CPU_ONLY
                               or not torch.backends.mps.is_available() else "mps")
            for seed in a.seeds:
                f = os.path.join(ck, f"{mn}_{ds}_{abl}_s{seed}.pt")
                if not os.path.exists(f):
                    continue
                todo = [v for v in VARIANTS if (ds, mn, seed, v) not in done]
                if not todo:
                    continue
                set_seed(seed)
                model = build(mn, abl, train_ds, dev, tslib)
                model.load_state_dict(torch.load(f, map_location=dev))
                model.eval()
                for v in todo:
                    pred, true = run(model, test_ds, dev, train_ds.mu,
                                     train_ds.sigma, train_ds.target_idx,
                                     VARIANTS[v], stride)
                    e = pred - true
                    row = {"model": mn, "dataset": ds, "ablation": abl,
                           "seed": seed, "variant": v,
                           "block_len": BLOCK, "n_windows": int(pred.shape[0]),
                           "test_stride": stride,
                           "MAE": float(np.abs(e).mean()),
                           "RMSE": float(np.sqrt((e ** 2).mean()))}
                    for c, n in enumerate(NAMES):
                        row[f"MAE_{n}"] = float(np.abs(e[:, :, c]).mean())
                        row[f"RMSE_{n}"] = float(np.sqrt((e[:, :, c] ** 2).mean()))
                    rows.append(row)
                    print(f"{mn:16s} {ds:22s} s{seed} {v:8s} "
                          f"MAE={row['MAE']:.4f}", flush=True)
                    pd.DataFrame(rows).to_csv(a.out, index=False)
                del model
    print(f"-> {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
