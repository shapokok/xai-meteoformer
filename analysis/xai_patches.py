"""Runtime patches that make GradientSHAP run on every baseline, fast.

Imported by the analysis harnesses only. Neither src/ nor the pinned
Time-Series-Library checkout is modified on disk.

1. Tensor.unfold on MPS -- see analysis/mps_unfold_fix.py. Lets PatchTST
   and XAI-MeteoFormer run on the GPU instead of CPU (34-117x faster).

2. In-place division in three TSLib baselines. PatchTST, iTransformer and
   TemporalFusionTransformer normalise with `x_enc /= stdev` inside
   forecast(). stdev is computed from x_enc, so autograd needs the original
   x_enc and raises "modified by an inplace operation" when asked for an
   input gradient. That is why GradientSHAP was missing for exactly these
   three models in the published fidelity table: src/xai.py caught the
   RuntimeError and silently fell back to permutation importance.

   The line is rewritten to `x_enc = x_enc / stdev`. The forward pass is
   BIT-IDENTICAL (checked with torch.equal on real checkpoints, CPU and
   MPS); only the autograd bookkeeping changes. TimesNet already uses the
   out-of-place `.div` and needs nothing.
"""

import inspect
import os
import sys
import textwrap
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import mps_unfold_fix  # noqa: E402,F401  (installs itself)

TSLIB = os.path.join(ROOT, "Time-Series-Library")
INPLACE = {"PatchTST": "x_enc /= stdev",
           "iTransformer": "x_enc /= stdev",
           "TemporalFusionTransformer": "x_enc /= stdev"}


def _patch_inplace():
    from baselines.tslib_adapter import _neutralize_reformer_import
    _neutralize_reformer_import(TSLIB)
    if TSLIB not in sys.path:
        sys.path.append(TSLIB)          # same position the adapter uses
    for name, needle in INPLACE.items():
        mod = __import__(f"models.{name}", fromlist=["Model"])
        cls = mod.Model
        src = textwrap.dedent(inspect.getsource(cls.forecast))
        n = src.count(needle)
        if n != 1:
            raise RuntimeError(f"{name}.forecast: expected 1 x '{needle}', "
                               f"found {n}; the pinned TSLib changed?")
        src = src.replace(needle, "x_enc = x_enc / stdev")
        ns = {}
        exec(compile(src, f"<patched {name}.forecast>", "exec"),
             vars(mod), ns)
        cls.forecast = ns["forecast"]


_patch_inplace()
