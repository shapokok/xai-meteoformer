"""Reversible Instance Normalization (RevIN).

Kim et al., ICLR 2022. On meteorological data this is the single largest
source of accuracy on transformer forecasters — do not disable it when
comparing against PatchTST / iTransformer, they both use it.

Difference from the reference implementation: `denormalize` accepts a
subset of channel indices, because the model predicts 4 targets out of
N input channels.
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True,
                 mode: str = "full"):
        """mode:
          "full"      - subtract the window mean AND divide by the window sd,
                        the standard RevIN.
          "mean_only" - subtract the window mean, do NOT divide. Isolates the
                        mechanism: analysis/error_decomposition.md shows the
                        RevIN penalty grows with the amplitude of the excursion
                        inside the forecast window, which is what dividing by
                        the sd of a quiet input window would cause. If the
                        damage comes from the scale term, mean_only should
                        recover most of the no_revin advantage; if it comes
                        from the centring, it should not.
        """
        super().__init__()
        assert mode in ("full", "mean_only"), mode
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.mode = mode
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        self.mean = None
        self.stdev = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) -> (B, L, C)"""
        self.mean = x.mean(dim=1, keepdim=True).detach()             # (B, 1, C)
        if self.mode == "full":
            self.stdev = torch.sqrt(
                x.var(dim=1, keepdim=True, unbiased=False) + self.eps
            ).detach()                                                # (B, 1, C)
        else:
            self.stdev = torch.ones_like(self.mean)
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.weight + self.bias
        return x

    def denormalize(self, y: torch.Tensor, channel_idx: torch.Tensor) -> torch.Tensor:
        """y: (B, H, n_targets) in normalized space -> original scale.

        channel_idx: LongTensor (n_targets,) — positions of the targets
        among the N input channels.
        """
        if self.mean is None:
            raise RuntimeError("normalize() must be called before denormalize()")
        if self.affine:
            w = self.weight[channel_idx].view(1, 1, -1)
            b = self.bias[channel_idx].view(1, 1, -1)
            y = (y - b) / (w + self.eps * self.eps)
        stdev = self.stdev[:, :, channel_idx]                          # (B, 1, n_targets)
        mean = self.mean[:, :, channel_idx]                            # (B, 1, n_targets)
        return y * stdev + mean


if __name__ == "__main__":
    B, L, C, H, T = 4, 96, 21, 24, 4
    layer = RevIN(C)
    x = torch.randn(B, L, C) * 5 + 12
    xn = layer.normalize(x)
    idx = torch.tensor([0, 1, 2, 3])
    y = torch.randn(B, H, T)
    yd = layer.denormalize(y, idx)
    assert xn.shape == (B, L, C), xn.shape
    assert yd.shape == (B, H, T), yd.shape
    # round-trip check on the target channels
    rt = layer.denormalize(xn[:, :H, idx], idx)
    assert torch.allclose(rt, x[:, :H, idx], atol=1e-3), (rt - x[:, :H, idx]).abs().max()
    print("revin ok:", xn.shape, yd.shape)
