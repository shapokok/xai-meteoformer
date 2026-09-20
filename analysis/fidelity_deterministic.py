"""Deterministic explanation fidelity -- the primary fidelity metric.

The fidelity metric in src/xai.py has two independent sources of Monte-Carlo
noise, which together give an sd of ~0.18 on a single fixed checkpoint
(analysis/fidelity_estimator_noise.csv): the perturbation (a random time
permutation of each perturbed channel) and the reference (ONE random channel
ordering). Both are replaced here; the definition is otherwise unchanged.

  perturbation  a channel is replaced by its own per-window mean: the level
                is kept, the temporal detail is destroyed. No randomness.
  reference     at k = 1 the exact expectation over random orderings (the mean
                single-channel occlusion); for k >= 2 the mean curve over
                M = 8 channel orderings drawn ONCE with a fixed seed and
                SHARED by every model and seed -- a fixed sample, not the exact
                expectation, but identical for every model, so it cannot
                favour one.
  rankings      occl  single-channel window-mean occlusion importance
                      (deterministic, model-agnostic; replaces permutation
                      importance, and like it is tuned to this test)
                shap  GradientSHAP -- the same function and budget as
                      src/xai.py (16 samples, 10 batches, 128 reference
                      windows) with torch seeded to 0 for every run
                attn  the proposed model's built-in variable attention
  scoring       ks = [0, 1, 2, 3, 5, 8]; gain = mean over k > 0 of
                (ranked curve - reference curve); rel = gain / base MAE.
                Identical to src/xai.py.

Same 2 560 test windows (first 40 batches of 64), physical channels only.
Baselines are explained in the variant the main table reports
(analysis/selection.py). Inference only, on existing checkpoints.

Output -> analysis/fidelity_det.csv, xai_det/*.npz
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
sys.path[:0] = [os.path.join(ROOT, "analysis"), os.path.join(ROOT, "src")]
import xai_patches  # noqa: E402,F401  MPS unfold + in-place autograd fixes
from data.dataset import build_splits                       # noqa: E402
from train import ABLATIONS                                 # noqa: E402
from xai import TIME_COLS, gradient_shap                    # noqa: E402
from xm_models.xai_meteoformer import XAIMeteoFormer        # noqa: E402
from baselines.tslib_adapter import build_baseline          # noqa: E402
from selection import selected_norm, selected_suffix        # noqa: E402

OURS = "XAI-MeteoFormer"
KS = [0, 1, 2, 3, 5, 8]
M_REF, REF_SEED = 8, 12345
MAX_BATCHES, BS, CHUNK = 40, 64, 1024
DEFAULTS = dict(seq_len=96, pred_len=24, patch_len=16, stride=8, d_model=256,
                n_heads=8, n_layers=2, dropout=0.2)


def occlude(x, chans):
    """replace each channel in `chans` by its own per-window mean"""
    if len(chans) == 0:
        return x
    x = x.clone()
    idx = list(chans)
    x[:, :, idx] = x[:, :, idx].mean(dim=1, keepdim=True)
    return x


@torch.no_grad()
def mae_many(model, variants, y):
    """MAE (scaled space, as src/xai.py) of each input variant, batched"""
    out, B = [], y.shape[0]
    stack = torch.cat(variants)
    preds = []
    for i in range(0, stack.shape[0], CHUNK):
        preds.append(model(stack[i:i + CHUNK])["y_pred"].float())
    preds = torch.cat(preds).view(len(variants), B, *y.shape[1:])
    return (preds - y.unsqueeze(0)).abs().mean(dim=(1, 2, 3)).cpu().numpy()


def build(model_name, abl, ds, train_ds, dev):
    from types import SimpleNamespace
    if model_name == OURS:
        return XAIMeteoFormer(n_channels=train_ds.n_channels,
                              target_idx=train_ds.target_idx, **DEFAULTS,
                              **ABLATIONS[abl]).to(dev)
    return build_baseline(model_name, SimpleNamespace(**DEFAULTS),
                          train_ds.n_channels, train_ds.target_idx,
                          train_ds.target_names,
                          tslib_path=os.path.join(ROOT, "Time-Series-Library"),
                          norm_variant=selected_norm(model_name, ds)).to(dev)


def explain(model, batches, keep, has_attn, dev):
    xs = [b["x"].to(dev) for b in batches]
    ys = [b["y"].to(dev) for b in batches]
    n_tot = sum(x.shape[0] for x in xs)

    # pass 1: base MAE and single-channel occlusion importance
    base, occl = 0.0, np.zeros(len(keep))
    for x, y in zip(xs, ys):
        m = mae_many(model, [x] + [occlude(x, [c]) for c in keep], y)
        base += m[0] * x.shape[0]
        occl += (m[1:] - m[0]) * x.shape[0]
    base /= n_tot
    occl /= n_tot

    # SHAP: same function and budget as src/xai.py, seeded identically
    torch.manual_seed(0)
    ref = torch.cat([b["x"] for b in batches[:2]])[:128].to(dev)
    acc = [gradient_shap(model, x, ref).detach().cpu().numpy()
           for x in xs[:max(1, MAX_BATCHES // 4)]]
    shap = np.concatenate(acc).mean(axis=0)[keep]

    attn = None
    if has_attn:
        with torch.no_grad():
            va = [model(x, return_explanations=True)["var_attn"].float().cpu().numpy()
                  for x in xs]
        attn = np.concatenate(va).mean(axis=(0, 1))[keep]

    # pass 2: curves. k = 0 is the base MAE for every ordering, and k = 1 is a
    # single-channel occlusion, already measured exactly in pass 1 -- including
    # the reference, whose k = 1 point is then the exact mean over channels
    # rather than a sample of orderings. Only k >= 2 needs new forward passes.
    rng = np.random.default_rng(REF_SEED)
    ref_orders = [list(np.asarray(keep)[rng.permutation(len(keep))])
                  for _ in range(M_REF)]
    rankings = {"occl": occl, "shap": shap}
    if attn is not None:
        rankings["attn"] = attn
    orders = {k: list(np.asarray(keep)[np.argsort(-v)]) for k, v in rankings.items()}
    ks = [k for k in KS if k <= len(keep)]
    single = dict(zip(keep, base + occl))           # MAE with channel c occluded
    curves = {k: np.zeros(len(ks)) for k in list(orders) + ["ref"]}
    for name in curves:
        curves[name][ks.index(0)] = base
        curves[name][ks.index(1)] = (float(np.mean(list(single.values())))
                                     if name == "ref" else single[orders[name][0]])
    big = [j for j, k in enumerate(ks) if k >= 2]
    for x, y in zip(xs, ys):
        variants, where = [], []
        for name, order in orders.items():
            for j in big:
                variants.append(occlude(x, order[:ks[j]])); where.append((name, j, 1.0))
        for o in ref_orders:
            for j in big:
                variants.append(occlude(x, o[:ks[j]])); where.append(("ref", j, 1.0 / M_REF))
        m = mae_many(model, variants, y)
        for (name, j, w), v in zip(where, m):
            curves[name][j] += w * v * x.shape[0] / n_tot

    nz = np.array(ks) > 0
    res = {"base_mae": base}
    for name in orders:
        gain = float((curves[name][nz] - curves["ref"][nz]).mean())
        res[f"det_gain_{name}"] = gain
        res[f"det_gain_{name}_rel"] = gain / max(base, 1e-9)
    res["det_agree_shap_occl_rho"] = spearmanr(shap, occl).statistic
    if attn is not None:
        res["det_agree_attn_shap_rho"] = spearmanr(attn, shap).statistic
        res["det_agree_attn_occl_rho"] = spearmanr(attn, occl).statistic
    return res, dict(occl=occl, shap=shap, attn=attn, ks=np.array(ks),
                     **{f"curve_{k}": v for k, v in curves.items()})


def jobs(datasets, seeds, ours_ablations):
    ck = os.path.join(ROOT, "checkpoints")
    out = []
    for ds in datasets:
        models = sorted({f.split(f"_{ds}_")[0] for f in os.listdir(ck)
                         if f.endswith(".pt") and f"_{ds}_" in f})
        for m in models:
            abls = ours_ablations if m == OURS else ["full"]
            for abl in abls:
                sfx = "" if m == OURS else selected_suffix(m, ds)
                for s in seeds:
                    f = os.path.join(ck, f"{m}_{ds}_{abl}{sfx}_s{s}.pt")
                    if os.path.exists(f):
                        out.append((ds, m, abl, sfx, s, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["jena", "beijing_aotizhongxin"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ours", nargs="*",
                    default=["no_revin", "no_revin+no_entropy", "full"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis", "fidelity_det.csv"))
    ap.add_argument("--npz_dir", default=os.path.join(ROOT, "xai_det"))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.npz_dir, exist_ok=True)
    dev = torch.device(a.device)
    rows = pd.read_csv(a.out).to_dict("records") if os.path.exists(a.out) else []
    done = {(r["dataset"], r["model"], r["ablation"], r["suffix"] if isinstance(r["suffix"], str) else "", r["seed"]) for r in rows}
    todo = [j for j in jobs(a.datasets, a.seeds, a.ours)
            if (not a.models or j[1] in a.models) and j[:5] not in done]
    print(f"{len(todo)} runs", flush=True)
    splits = {}
    for i, (ds, m, abl, sfx, s, f) in enumerate(todo, 1):
        if ds not in splits:
            tr, _, te = build_splits(os.path.join(ROOT, "data/processed"), ds,
                                     seq_len=96, pred_len=24, seed=0)
            it = iter(DataLoader(te, batch_size=BS, shuffle=False))
            splits[ds] = (tr, [next(it) for _ in range(MAX_BATCHES)],
                          [c for c, nme in enumerate(te.columns)
                           if nme not in TIME_COLS])
        tr, batches, keep = splits[ds]
        model = build(m, abl, ds, tr, dev)
        model.load_state_dict(torch.load(f, map_location=dev))
        model.eval()
        res, arr = explain(model, batches, keep, m == OURS, dev)
        tag = os.path.basename(f)[:-3]
        np.savez(os.path.join(a.npz_dir, f"{tag}_det.npz"),
                 **{k: v for k, v in arr.items() if v is not None})
        rows.append({"dataset": ds, "model": m, "ablation": abl, "suffix": sfx,
                     "seed": s, **res})
        pd.DataFrame(rows).to_csv(a.out, index=False)
        rel = {k[9:-4]: round(v, 3) for k, v in res.items() if k.endswith("_rel")}
        print(f"[{i}/{len(todo)}] {tag:50s} {rel}", flush=True)
        del model
        if dev.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
