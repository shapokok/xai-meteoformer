"""Explanation extraction and validation — the paper's main contribution.

Produces, for a trained checkpoint:

  variable importance   from the model's own single-head variable attention
                        (ours only), from GradientSHAP, and from permutation
                        importance (works for every model, so baselines are
                        comparable)
  temporal importance   interpretable pooling weights and attention rollout
  fidelity              permute the top-k variables, measure the error rise.
                        Permutation WITHIN the window, never mean-imputation:
                        replacing a channel by its mean pushes the input off
                        the training manifold, and part of the error rise then
                        measures that shift rather than the variable's
                        importance (Hooker et al., ROAR).
  stability             re-explain under input noise, compare rankings by
                        Spearman / cosine / top-k overlap
  agreement             Spearman between attention and SHAP — the direct answer
                        to "attention is not explanation"
  horizon-specific      importance recomputed at h = 1, 6, 12, 24

    python src/xai.py --dataset jena --ckpt outputs/checkpoints/XAI-MeteoFormer_jena_full_s0.pt
    python src/xai.py --dataset jena --model PatchTST --ckpt ... --skip_internal

Writes outputs/xai/<tag>_*.npz and appends a row per run to
outputs/xai_metrics.csv.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines.tslib_adapter import build_baseline          # noqa: E402
from data.dataset import build_splits                        # noqa: E402
from xm_models.xai_meteoformer import XAIMeteoFormer         # noqa: E402

REPORT_HORIZONS = [1, 6, 12, 24]

# Calendar encodings are legitimate inputs and genuinely predictive — day of
# year carries seasonal climatology. But if they rank near the top, the
# explanation is partly "the model knows what season it is", which is not the
# physical claim the paper makes. --exclude_time reruns the ranking over
# physical channels only, so both readings can be reported side by side.
TIME_COLS = ("hour_sin", "hour_cos", "doy_sin", "doy_cos")


# --------------------------------------------------------------------------- #
def gradient_shap(model, x, baselines, n_samples=16, stdev=0.09):
    """GradientSHAP attributions, (B, L, N) -> (B, N).

    KernelSHAP is not an option here: it needs thousands of forward passes
    per sample, which on a transformer over 96x19 inputs does not finish.
    """
    total = torch.zeros_like(x)
    for _ in range(n_samples):
        idx = torch.randint(0, baselines.shape[0], (x.shape[0],),
                            device=x.device)
        base = baselines[idx]
        alpha = torch.rand(x.shape[0], 1, 1, device=x.device)
        noise = torch.randn_like(x) * stdev
        inp = (base + alpha * (x - base) + noise).detach().requires_grad_(True)
        out = model(inp)["y_pred"].sum()
        g, = torch.autograd.grad(out, inp)
        total = total + g.detach() * (x - base)
    attr = (total / n_samples).abs().sum(dim=1)              # (B, N)
    return attr


@torch.no_grad()
def batch_mae(model, x, y):
    return (model(x)["y_pred"] - y).abs().mean().item()


@torch.no_grad()
def permutation_importance(model, loader, device, n_channels, rng):
    """Model-agnostic importance: shuffle one channel along time inside each
    window, measure the MAE rise. Applies to every baseline too."""
    base, imp, n = 0.0, np.zeros(n_channels), 0
    for b in loader:
        x = b["x"].to(device)
        y = b["y"].to(device)
        base += batch_mae(model, x, y) * x.size(0)
        for c in range(n_channels):
            xp = x.clone()
            perm = torch.from_numpy(
                rng.permutation(x.shape[1])).to(device)
            xp[:, :, c] = x[:, perm, c]
            imp[c] += batch_mae(model, xp, y) * x.size(0)
        n += x.size(0)
    base /= n
    return imp / n - base, base


@torch.no_grad()
def fidelity_curve(model, loader, device, order, ks, rng):
    """Permute the k most important channels; MAE should rise fastest when the
    ranking is right. A flat curve means the explanation is not faithful."""
    out = []
    for k in ks:
        tot, n = 0.0, 0
        for b in loader:
            x = b["x"].to(device)
            y = b["y"].to(device)
            xp = x.clone()
            for c in order[:k]:
                perm = torch.from_numpy(rng.permutation(x.shape[1])).to(device)
                xp[:, :, c] = x[:, perm, c]
            tot += batch_mae(model, xp, y) * x.size(0)
            n += x.size(0)
        out.append(tot / n)
    return np.array(out)


def rank_agreement(a, b, top_k=5):
    rho = spearmanr(a, b).correlation
    cos = float(np.dot(a, b) /
                (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    ta, tb = set(np.argsort(-a)[:top_k]), set(np.argsort(-b)[:top_k])
    return rho, cos, len(ta & tb) / top_k


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", default="XAI-MeteoFormer")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--skip_internal", action="store_true",
                   help="baselines have no built-in attention")
    p.add_argument("--max_batches", type=int, default=40,
                   help="explanations are averaged over this many test batches")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=24)
    p.add_argument("--patch_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--noise_levels", type=float, nargs="*",
                   default=[0.01, 0.05, 0.1])
    p.add_argument("--exclude_time", action="store_true",
                   help="rank and score physical channels only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tslib_path", default=None)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    _, _, test = build_splits(args.processed_dir, args.dataset,
                              seq_len=args.seq_len, pred_len=args.pred_len)
    names = test.columns
    N = test.n_channels
    loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

    # a fixed, bounded slice of the test set — explanations are averaged, so
    # more batches change the numbers only marginally but cost linearly
    batches = []
    for i, b in enumerate(loader):
        if i >= args.max_batches:
            break
        batches.append({k: v.to(device) for k, v in b.items()})

    if args.model == "XAI-MeteoFormer":
        model = XAIMeteoFormer(
            n_channels=N, target_idx=test.target_idx,
            seq_len=args.seq_len, pred_len=args.pred_len,
            patch_len=args.patch_len, stride=args.stride,
            d_model=args.d_model, n_heads=args.n_heads,
            n_layers=args.n_layers, dropout=args.dropout).to(device)
    else:
        model = build_baseline(args.model, args, N, test.target_idx,
                               test.target_names,
                               tslib_path=args.tslib_path).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    res = {"model": args.model, "dataset": args.dataset,
           "ckpt": os.path.basename(args.ckpt)}
    store = {"channels": np.array(names)}

    # ---- 1. built-in attention ------------------------------------------- #
    if not args.skip_internal:
        var_a, temp_a, roll, gates = [], [], [], []
        with torch.no_grad():
            for b in batches:
                o = model(b["x"], return_explanations=True)
                var_a.append(o["var_attn"].cpu().numpy())
                temp_a.append(o["temp_attn"].cpu().numpy())
                roll.append(o["rollout"].cpu().numpy())
                if "scale_gates" in o:
                    gates.append(o["scale_gates"].cpu().numpy())
        var_attn = np.concatenate(var_a)                 # (n, n_targets, N)
        store["var_attn"] = var_attn
        store["temp_attn"] = np.concatenate(temp_a)
        store["rollout"] = np.concatenate(roll)
        if gates:
            store["scale_gates"] = np.concatenate(gates)
        attn_imp = var_attn.mean(axis=(0, 1))            # (N,)
        store["attn_importance"] = attn_imp
        res["attn_entropy"] = float(
            -(var_attn * np.log(var_attn + 1e-9)).sum(-1).mean())
    else:
        attn_imp = None

    # ---- 2. GradientSHAP -------------------------------------------------- #
    ref = torch.cat([b["x"] for b in batches[:2]])[:128]
    shap_acc = []
    for b in batches[:max(1, args.max_batches // 4)]:
        shap_acc.append(gradient_shap(model, b["x"], ref).detach().cpu().numpy())
    shap_imp = np.concatenate(shap_acc).mean(axis=0)     # (N,)
    store["shap_importance"] = shap_imp

    # ---- 3. permutation importance (model-agnostic) ----------------------- #
    perm_imp, base_mae = permutation_importance(model, batches, device, N, rng)
    store["perm_importance"] = perm_imp
    res["base_mae"] = base_mae

    # ---- 4. fidelity ------------------------------------------------------ #
    keep = np.array([i for i, c in enumerate(names)
                     if not (args.exclude_time and c in TIME_COLS)])
    res["n_channels_scored"] = int(len(keep))
    res["exclude_time"] = bool(args.exclude_time)

    ks = [0, 1, 2, 3, 5, 8]
    ks = [k for k in ks if k <= len(keep)]
    ref_imp = attn_imp if attn_imp is not None else shap_imp
    order = list(keep[np.argsort(-ref_imp[keep])])
    rand_order = list(rng.permutation(keep))
    store["fidelity_ks"] = np.array(ks)
    store["fidelity_explained"] = fidelity_curve(model, batches, device,
                                                 order, ks, rng)
    store["fidelity_random"] = fidelity_curve(model, batches, device,
                                              rand_order, ks, rng)
    # area between the two curves: how much better than a random ranking
    nz = np.array(ks) > 0
    res["fidelity_gain"] = float(
        (store["fidelity_explained"][nz] - store["fidelity_random"][nz]).mean())
    res["fidelity_gain_rel"] = res["fidelity_gain"] / max(base_mae, 1e-9)

    # ---- 5. stability ----------------------------------------------------- #
    for s in args.noise_levels:
        noisy = []
        with torch.no_grad():
            for b in batches:
                xn = b["x"] + torch.randn_like(b["x"]) * s
                if args.skip_internal:
                    continue
                noisy.append(model(xn, return_explanations=True)
                             ["var_attn"].cpu().numpy())
        if noisy:
            imp_n = np.concatenate(noisy).mean(axis=(0, 1))
            rho, cos, ov = rank_agreement(ref_imp, imp_n)
            res[f"stab_rho_{s}"] = rho
            res[f"stab_cos_{s}"] = cos
            res[f"stab_top5_{s}"] = ov
            store[f"importance_noise_{s}"] = imp_n

    # ---- 6. agreement attention vs SHAP vs permutation -------------------- #
    if attn_imp is not None:
        r, c, o = rank_agreement(attn_imp[keep], shap_imp[keep])
        res["agree_attn_shap_rho"] = r
        res["agree_attn_shap_cos"] = c
        res["agree_attn_shap_top5"] = o
        r, c, o = rank_agreement(attn_imp[keep], perm_imp[keep])
        res["agree_attn_perm_rho"] = r
        res["agree_attn_perm_top5"] = o
    r, c, o = rank_agreement(shap_imp[keep], perm_imp[keep])
    res["agree_shap_perm_rho"] = r
    res["agree_shap_perm_top5"] = o

    # ---- 7. horizon-specific importance ----------------------------------- #
    if not args.skip_internal:
        hz = {}
        for h in REPORT_HORIZONS:
            if h > args.pred_len:
                continue
            acc = []
            for b in batches[:8]:
                x = b["x"].detach().requires_grad_(True)
                out = model(x)["y_pred"][:, h - 1, :].sum()
                g, = torch.autograd.grad(out, x)
                acc.append(g.detach().abs().sum(1).cpu().numpy())
            hz[str(h)] = np.concatenate(acc).mean(axis=0)
        store["horizon_importance"] = np.stack([hz[k] for k in sorted(hz, key=int)])
        store["horizon_steps"] = np.array(sorted(map(int, hz)))

    # ---- save -------------------------------------------------------------- #
    xdir = os.path.join(args.out_dir, "xai")
    os.makedirs(xdir, exist_ok=True)
    tag = os.path.basename(args.ckpt).replace(".pt", "")
    np.savez_compressed(os.path.join(xdir, f"{tag}_xai.npz"), **store)

    csv = os.path.join(args.out_dir, "xai_metrics.csv")
    pd.DataFrame([res]).to_csv(csv, mode="a", header=not os.path.exists(csv),
                               index=False)

    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in res.items()}, indent=2))
    top = keep[np.argsort(-ref_imp[keep])][:8]
    print("\ntop variables:", ", ".join(
        f"{names[i]} ({ref_imp[i]:.3f})" for i in top))
    print(f"saved {xdir}/{tag}_xai.npz  and appended to {csv}")


if __name__ == "__main__":
    main()
