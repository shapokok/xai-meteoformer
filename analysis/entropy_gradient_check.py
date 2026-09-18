"""Does the entropy term actually influence training?

Three checks, all on CPU, no training:
  1. autograd: does d(var_entropy)/d(theta) reach the parameters at all?
  2. scale: how big is lambda_ent * entropy next to the Huber term?
  3. evidence from the trained weights: do the full (lambda_ent=0.01) and
     no_entropy (lambda_ent=0) checkpoints actually differ, and is the
     attention measurably sparser in the former?
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data.dataset import build_splits                  # noqa: E402
from xm_models.xai_meteoformer import XAIMeteoFormer   # noqa: E402

torch.manual_seed(0)
dev = torch.device("cpu")
train_ds, val_ds, test_ds = build_splits(os.path.join(ROOT, "data/processed"),
                                         "jena", seq_len=96, pred_len=24, seed=0)


def make(**kw):
    return XAIMeteoFormer(n_channels=train_ds.n_channels,
                          target_idx=train_ds.target_idx, seq_len=96,
                          pred_len=24, patch_len=16, stride=8, d_model=256,
                          n_heads=8, n_layers=2, dropout=0.2, **kw).to(dev)


print("=" * 72)
print("1. autograd reachability of var_entropy")
print("=" * 72)
for name, kw in [("use_var_attn=True (full)", {}),
                 ("use_var_attn=False (no_var_attn)", {"use_var_attn": False})]:
    m = make(**kw)
    m.train()
    b = next(iter(DataLoader(train_ds, batch_size=32, shuffle=False)))
    out = m(b["x"].to(dev))
    ent = out["var_entropy"]
    m.zero_grad(set_to_none=True)
    if ent.requires_grad:
        ent.backward()
        gr = {n: p.grad.abs().sum().item() for n, p in m.named_parameters()
              if p.grad is not None and p.grad.abs().sum() > 0}
        tot = sum(gr.values())
        print(f"\n{name}")
        print(f"  entropy value            : {ent.item():.6f}  "
              f"(max possible ln(N)=ln({train_ds.n_channels})="
              f"{np.log(train_ds.n_channels):.4f})")
        print(f"  requires_grad            : {ent.requires_grad}")
        print(f"  params receiving nonzero grad: {len(gr)} of "
              f"{sum(1 for _ in m.parameters())}")
        print(f"  total |grad|             : {tot:.6g}")
        for n_, v in sorted(gr.items(), key=lambda kv: -kv[1])[:5]:
            print(f"    {n_:45s} {v:.6g}")
    else:
        print(f"\n{name}")
        print(f"  entropy value            : {ent.item():.6f}")
        print(f"  requires_grad            : False  -> NO gradient path")

print()
print("=" * 72)
print("2. magnitude of the entropy term against the regression term")
print("=" * 72)
m = make()
m.train()
reg = nn.HuberLoss(delta=1.0)
b = next(iter(DataLoader(train_ds, batch_size=64, shuffle=False)))
out = m(b["x"].to(dev))
h = reg(out["y_pred"], b["y"].to(dev))
e = out["var_entropy"]
for lam in (0.0005, 0.01):
    print(f"  Huber={h.item():.5f}   lambda_ent={lam}: "
          f"lam*entropy={lam * e.item():.5f}  "
          f"({100 * lam * e.item() / (h.item() + lam * e.item()):.2f}% of total)")

print()
print("=" * 72)
print("3. trained checkpoints: full (lambda_ent=0.01) vs no_entropy (0.0)")
print("=" * 72)
ck = os.path.join(ROOT, "checkpoints")
rows = []
for seed in range(3):
    sds = {}
    for abl in ("full", "no_entropy"):
        f = os.path.join(ck, f"XAI-MeteoFormer_jena_{abl}_s{seed}.pt")
        if not os.path.exists(f):
            sds = None
            break
        sds[abl] = torch.load(f, map_location="cpu")
    if sds is None:
        print(f"  seed {seed}: checkpoint missing, skipped")
        continue
    a, bb = sds["full"], sds["no_entropy"]
    q = "variable_query"
    same = all(torch.equal(a[k], bb[k]) for k in a)
    dq = (a[q] - bb[q]).abs().max().item() if q in a else float("nan")
    # realised attention entropy on a val batch
    ents = {}
    for abl, sd in sds.items():
        mm = make()
        mm.load_state_dict(sd)
        mm.eval()
        with torch.no_grad():
            vb = next(iter(DataLoader(val_ds, batch_size=256, shuffle=False)))
            o = mm(vb["x"], return_explanations=True)
            beta = o["var_attn"]
            ents[abl] = (-(beta * (beta + 1e-9).log()).sum(-1)).mean().item()
    rows.append((seed, same, dq, ents["full"], ents["no_entropy"]))
    print(f"  seed {seed}: identical weights={same}  "
          f"max|d variable_query|={dq:.4g}  "
          f"val entropy: full={ents['full']:.4f}  "
          f"no_entropy={ents['no_entropy']:.4f}")
if rows:
    f_ = np.mean([r[3] for r in rows]); n_ = np.mean([r[4] for r in rows])
    print(f"\n  mean realised entropy: full={f_:.4f}  no_entropy={n_:.4f}  "
          f"delta={f_ - n_:+.4f}   (ln(N)={np.log(train_ds.n_channels):.4f})")
