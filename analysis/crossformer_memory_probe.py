"""Estimate Crossformer's training memory at different widths.

No CUDA is available on this machine, so peak VRAM cannot be measured
directly. What CAN be measured exactly is the parameter count and the
total size of the forward activations, which dominate. Activations are
summed with forward hooks over every module output; training memory is
then

    params (fp32) + grads + 2 AdamW moments   = 4 x params x 4 bytes
  + activations retained for backward          ~ 1x the measured sum
  + a workspace allowance

The same script, run on the T4, prints the true figure via
torch.cuda.max_memory_allocated -- see the notebook cell it emits.
"""

import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from baselines.tslib_adapter import build_baseline, MODEL_OVERRIDES  # noqa: E402


def measure(d_model, d_ff, batch, n_channels=19, seq_len=96):
    args = SimpleNamespace(seq_len=seq_len, pred_len=24, d_model=d_model,
                           n_heads=8, n_layers=2, dropout=0.2,
                           patch_len=16, stride=8)
    ov = dict(MODEL_OVERRIDES["Crossformer"])
    ov.update(d_model=d_model, d_ff=d_ff)
    saved = MODEL_OVERRIDES["Crossformer"]
    MODEL_OVERRIDES["Crossformer"] = ov
    try:
        m = build_baseline("Crossformer", args, n_channels, [0, 1, 2, 3],
                           ["T", "RH", "P", "WS"],
                           tslib_path=os.path.join(ROOT, "Time-Series-Library"),
                           norm_variant="off")
    finally:
        MODEL_OVERRIDES["Crossformer"] = saved

    total = [0]
    hooks = []

    def hook(_mod, _inp, out):
        for t in (out if isinstance(out, (tuple, list)) else [out]):
            if torch.is_tensor(t):
                total[0] += t.numel() * t.element_size()

    for mod in m.modules():
        hooks.append(mod.register_forward_hook(hook))
    x = torch.randn(batch, seq_len, n_channels)
    m(x)
    for h in hooks:
        h.remove()

    params = sum(p.numel() for p in m.parameters())
    MB = 1024 ** 2
    p_mb = params * 4 / MB
    a_mb = total[0] / MB
    train_mb = 4 * p_mb + 2 * a_mb          # params+grad+2 moments, acts + grads
    return params, p_mb, a_mb, train_mb


def main():
    print(f"{'config':34s} {'params':>11s} {'param MB':>9s} "
          f"{'act MB':>9s} {'est train MB':>13s}")
    rows = []
    for dm, dff, bs in [(128, 128, 64), (128, 128, 128),
                        (256, 1024, 64), (256, 1024, 128),
                        (256, 512, 64)]:
        try:
            p, pmb, amb, tmb = measure(dm, dff, bs)
            tag = f"d_model={dm} d_ff={dff} batch={bs}"
            print(f"{tag:34s} {p:11,d} {pmb:9.1f} {amb:9.1f} {tmb:13.1f}")
            rows.append((dm, dff, bs, p, pmb, amb, tmb))
        except Exception as e:
            print(f"d_model={dm} d_ff={dff} batch={bs}: FAILED {type(e).__name__}: {e}")
    print()
    print("T4 has 15360 MiB usable. AMP roughly halves the activation term.")
    return rows


if __name__ == "__main__":
    main()
