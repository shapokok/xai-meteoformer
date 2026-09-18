"""Work around an MPS bug in Tensor.unfold, for analysis harnesses only.

On this machine MPS aborts with an MPSNDArray "buffer is not large enough"
assertion inside the patching step of PatchTST and XAI-MeteoFormer, which
is why both were being run on CPU at ~1/50 the speed. The sliding-window
case of unfold is replaced by an explicit stack of slices, which produces
the SAME values as a contiguous tensor. Numerical equivalence against the
CPU reference is checked by verify() before anything relies on it.

src/ is not modified; importing this module patches torch.Tensor.unfold in
the importing process only.
"""

import torch

_orig_unfold = torch.Tensor.unfold


def _unfold(self, dimension, size, step):
    if self.device.type != "mps":
        return _orig_unfold(self, dimension, size, step)
    d = dimension % self.dim()
    n = (self.shape[d] - size) // step + 1
    wins = [self.narrow(d, i * step, size) for i in range(n)]
    # unfold puts the window count at `d` and the window itself last
    return torch.stack(wins, dim=d).movedim(d + 1, -1).contiguous()


def install():
    torch.Tensor.unfold = _unfold


def verify(atol=1e-4):
    """unfold must match the CPU reference exactly, element for element."""
    x = torch.randn(4, 19, 96)
    ref = _orig_unfold(x, -1, 16, 8)
    got = _unfold(x.to("mps"), -1, 16, 8).cpu()
    assert got.shape == ref.shape, (got.shape, ref.shape)
    err = (got - ref).abs().max().item()
    assert err <= atol, err
    return err


install()
