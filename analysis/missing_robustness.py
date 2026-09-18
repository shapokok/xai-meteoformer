"""Robustness to synthetic gaps in the test input windows (Reviewer 1, Minor #7).

Uses the mechanism already in the codebase -- MeteoWindowDataset.missing_rate
-- so nothing new is invented for the experiment:

  * a Bernoulli(missing_rate) mask is drawn INDEPENDENTLY PER CELL of the
    (96, n_channels) input window, i.e. per timestep and channel, not per
    whole timestep;
  * masked cells are set to NaN and then filled by LINEAR INTERPOLATION
    along time within the window (np.interp over the valid indices of that
    channel); if a channel is entirely missing it is set to 0.0, which is
    the mean in scaled space;
  * the mask is applied to the SCALED input only. Targets are never masked,
    so the metric is comparable to the clean run.
  * the RNG is seeded per dataset instance from the run's seed, so the gap
    pattern is reproducible and identical across models for a given seed.

Inference only, on the existing checkpoints.

Cost control. The quantity of interest is the DEGRADATION of a model
between missing rates, which is a within-model difference. A systematic
1-in-k subsample of the test windows estimates it to far more precision
than the effect size requires, at 1/k the cost. All four rates -- the
clean r=0 included -- are computed on the SAME subsample, so the deltas
are internally exact. The subsample is validated against full-test runs
in analysis/missing_robustness_fulltest.csv.

Output -> analysis/missing_robustness_sub.csv  (subsampled, all models)
          analysis/missing_robustness_fulltest.csv  (full test, partial)
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
from data.dataset import build_splits                      # noqa: E402
from train import ABLATIONS, set_seed                      # noqa: E402
from xm_models.xai_meteoformer import XAIMeteoFormer       # noqa: E402
from baselines.tslib_adapter import build_baseline         # noqa: E402

DEFAULTS = dict(seq_len=96, pred_len=24, patch_len=16, stride=8, d_model=256,
                n_heads=8, n_layers=2, dropout=0.2)
NAMES = ["T", "RH", "P", "WS"]
RATES = [0.0, 0.05, 0.10, 0.20]
# MPS raises an internal Metal assertion on Tensor.unfold (patching)
CPU_ONLY = {"PatchTST", "XAI-MeteoFormer"}


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
def run(model, ds, device, mu, sigma, tidx, bs=128, stride=1):
    view = ds if stride <= 1 else Subset(ds, list(range(0, len(ds), stride)))
    loader = DataLoader(view, batch_size=bs, shuffle=False, num_workers=0)
    P, T = [], []
    for b in loader:
        P.append(model(b["x"].to(device))["y_pred"].float().cpu().numpy())
        T.append(b["y_raw"].numpy())
    pred = np.concatenate(P) * sigma[tidx].reshape(1, 1, -1) + mu[tidx].reshape(1, 1, -1)
    return pred, np.concatenate(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["jena", "beijing_aotizhongxin"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--target_windows", type=int, default=2800,
                    help="subsample the test split to about this many windows; "
                         "0 disables subsampling")
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis",
                                                  "missing_robustness_sub.csv"))
    a = ap.parse_args()
    tslib = os.path.join(ROOT, "Time-Series-Library")
    ck = os.path.join(ROOT, "checkpoints")

    done = set()
    if os.path.exists(a.out):
        old = pd.read_csv(a.out)
        done = {(r.model, r.dataset, r.ablation, r.seed, r.missing_rate)
                for r in old.itertuples()}
        rows = old.to_dict("records")
    else:
        rows = []

    for ds in a.datasets:
        train_ds, _, _ = build_splits(os.path.join(ROOT, "data/processed"), ds,
                                      seq_len=96, pred_len=24, seed=0)
        models = a.models or sorted({f.split(f"_{ds}_")[0]
                                     for f in os.listdir(ck)
                                     if f.endswith(".pt") and f"_{ds}_" in f})
        for mn in models:
            abl = "no_revin" if mn == "XAI-MeteoFormer" else "full"
            dev = torch.device("cpu" if mn in CPU_ONLY or
                               not torch.backends.mps.is_available() else "mps")
            for seed in a.seeds:
                f = os.path.join(ck, f"{mn}_{ds}_{abl}_s{seed}.pt")
                if not os.path.exists(f):
                    continue
                todo = [r for r in RATES
                        if (mn, ds, abl, seed, r) not in done]
                if not todo:
                    continue
                set_seed(seed)
                model = build(mn, abl, train_ds, dev, tslib)
                model.load_state_dict(torch.load(f, map_location=dev))
                model.eval()
                for rate in todo:
                    # fresh dataset so the RNG state is reproducible per rate
                    _, _, test_ds = build_splits(
                        os.path.join(ROOT, "data/processed"), ds,
                        seq_len=96, pred_len=24, seed=seed)
                    test_ds.missing_rate = rate
                    stride = (max(1, round(len(test_ds) / a.target_windows))
                              if a.target_windows else 1)
                    pred, true = run(model, test_ds, dev, train_ds.mu,
                                     train_ds.sigma, train_ds.target_idx,
                                     stride=stride)
                    e = pred - true
                    row = {"model": mn, "dataset": ds, "ablation": abl,
                           "seed": seed, "missing_rate": rate,
                           "n_windows": int(pred.shape[0]),
                           "test_stride": stride,
                           "MAE": float(np.abs(e).mean()),
                           "RMSE": float(np.sqrt((e ** 2).mean()))}
                    for c, n in enumerate(NAMES):
                        row[f"MAE_{n}"] = float(np.abs(e[:, :, c]).mean())
                        row[f"RMSE_{n}"] = float(np.sqrt((e[:, :, c] ** 2).mean()))
                    rows.append(row)
                    print(f"{mn:16s} {ds:22s} s{seed} r={rate:.2f} "
                          f"MAE={row['MAE']:.4f} RMSE={row['RMSE']:.4f}", flush=True)
                    pd.DataFrame(rows).to_csv(a.out, index=False)
                del model
    print(f"-> {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
