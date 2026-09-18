"""External validation of the temporal-importance axis (Reviewer 1, Major #3/#4/#7).

The paper currently justifies its temporal explanation with the model's
own attention weights and with attention rollout. Neither is an external
check: both are read out of the same computation they claim to explain.

This script supplies the missing ground truth. For each input patch
position p (the same patches the model forms, patch_len=16, stride=8):

    occlude patch p in every input channel, re-run the model, and record
    the rise in test MAE.

Occlusion replaces the patch with the per-channel mean of the *rest of
that window*, so the window keeps its own level and only the local
detail is destroyed. That is the standard "uninformative replacement"
choice; zeroing would inject a spurious level shift in scaled space.

The resulting delta-MAE vector over patches is the external importance
ranking. It is compared, by Spearman rank correlation, against:
  * `temp_attn` - the interpretable pooling weights alpha
  * `rollout`   - attention rollout over the encoder layers

A high correlation means the attention is telling the truth about time.
A low one means it is not, and the paper's temporal claims rest on
nothing external.

Inference only, on existing checkpoints.

Output -> analysis/occlusion_time.csv, analysis/occlusion_time.md
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data.dataset import build_splits                      # noqa: E402
from train import ABLATIONS, set_seed                      # noqa: E402
from xm_models.xai_meteoformer import XAIMeteoFormer       # noqa: E402

SEQ, PRED, PATCH, STRIDE = 96, 24, 16, 8
N_PATCHES = (SEQ - PATCH) // STRIDE + 1          # 11


def patch_span(p):
    s = p * STRIDE
    return s, min(s + PATCH, SEQ)


@torch.no_grad()
def occlusion(model, loader, device, mu, sigma, tidx, max_batches):
    """-> (base_mae, delta_mae per patch (P,))"""
    base, deltas, n = 0.0, np.zeros(N_PATCHES), 0
    s = sigma[tidx].reshape(1, 1, -1)
    m = mu[tidx].reshape(1, 1, -1)
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        x = b["x"].to(device)
        y = b["y_raw"].numpy()
        bs = x.shape[0]
        p0 = model(x)["y_pred"].float().cpu().numpy() * s + m
        base += np.abs(p0 - y).mean() * bs
        for p in range(N_PATCHES):
            lo, hi = patch_span(p)
            xo = x.clone()
            # mean of the rest of the window, per sample and channel
            keep = torch.ones(SEQ, dtype=torch.bool, device=device)
            keep[lo:hi] = False
            rest = x[:, keep, :].mean(dim=1, keepdim=True)     # (B,1,C)
            xo[:, lo:hi, :] = rest
            pp = model(xo)["y_pred"].float().cpu().numpy() * s + m
            deltas[p] += (np.abs(pp - y).mean() - np.abs(p0 - y).mean()) * bs
        n += bs
    return base / n, deltas / n


@torch.no_grad()
def internal_rankings(model, loader, device, max_batches):
    """mean temp_attn and rollout over patches, averaged over channels"""
    ta, ro, n = np.zeros(N_PATCHES), np.zeros(N_PATCHES), 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        o = model(b["x"].to(device), return_explanations=True)
        bs = b["x"].shape[0]
        ta += o["temp_attn"].float().cpu().numpy().mean(axis=1).sum(axis=0)
        ro += o["rollout"].float().cpu().numpy().mean(axis=1).sum(axis=0)
        n += bs
    return ta / n, ro / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=["jena", "beijing_aotizhongxin"])
    ap.add_argument("--ablations", nargs="*", default=["no_revin", "no_entropy"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--max_batches", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    dst = os.path.join(ROOT, "analysis", "occlusion_time.csv")
    rows = pd.read_csv(dst).to_dict("records") if os.path.exists(dst) else []
    done = {(r["dataset"], r["ablation"], r["seed"]) for r in rows}
    dev = torch.device(a.device)

    for ds in a.datasets:
        train_ds, _, test_ds = build_splits(
            os.path.join(ROOT, "data/processed"), ds,
            seq_len=SEQ, pred_len=PRED, seed=0)
        for abl in a.ablations:
            for seed in a.seeds:
                if (ds, abl, seed) in done:
                    continue
                f = os.path.join(ROOT, "checkpoints",
                                 f"XAI-MeteoFormer_{ds}_{abl}_s{seed}.pt")
                if not os.path.exists(f):
                    continue
                set_seed(seed)
                model = XAIMeteoFormer(
                    n_channels=train_ds.n_channels,
                    target_idx=train_ds.target_idx, seq_len=SEQ, pred_len=PRED,
                    patch_len=PATCH, stride=STRIDE, d_model=256, n_heads=8,
                    n_layers=2, dropout=0.2, **ABLATIONS[abl]).to(dev)
                model.load_state_dict(torch.load(f, map_location=dev))
                model.eval()
                loader = DataLoader(test_ds, batch_size=a.batch_size,
                                    shuffle=False, num_workers=0)
                base, dmae = occlusion(model, loader, dev, train_ds.mu,
                                       train_ds.sigma, train_ds.target_idx,
                                       a.max_batches)
                ta, ro = internal_rankings(model, loader, dev, a.max_batches)
                r_ta = spearmanr(dmae, ta).statistic
                r_ro = spearmanr(dmae, ro).statistic
                r_rec = spearmanr(dmae, np.arange(N_PATCHES)).statistic
                row = {"dataset": ds, "ablation": abl, "seed": seed,
                       "base_MAE": base, "rho_temp_attn": r_ta,
                       "rho_rollout": r_ro, "rho_recency": r_rec}
                for p in range(N_PATCHES):
                    lo, hi = patch_span(p)
                    row[f"dmae_p{p}"] = dmae[p]
                    row[f"attn_p{p}"] = ta[p]
                    row[f"roll_p{p}"] = ro[p]
                    row[f"span_p{p}"] = f"{lo}-{hi}"
                rows.append(row)
                pd.DataFrame(rows).to_csv(dst, index=False)
                print(f"{ds:22s} {abl:10s} s{seed} base={base:.4f} "
                      f"rho(attn)={r_ta:+.3f} rho(rollout)={r_ro:+.3f} "
                      f"rho(recency)={r_rec:+.3f}", flush=True)
                del model
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
