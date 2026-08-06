"""XAI-MeteoFormer.

RevIN -> multi-scale dilated block -> patching -> transformer encoder
     -> dual explainable attention (temporal + variable)
     -> adaptive fusion -> regression head + threshold-event head

Design notes that matter for the paper:

* Variable attention is deliberately SINGLE-HEAD. Raw multi-head weights
  are not a defensible explanation (Jain & Wallace 2019); one head gives
  an unambiguous importance vector that can be compared against SHAP.
* Temporal importance is exposed two ways: the interpretable pooling
  weights (alpha) and attention rollout over the encoder layers. Report
  rollout in the paper, keep alpha as the ablation-friendly variant.
* The entropy of the variable attention is returned so the training loop
  can add it to the loss. Minimizing it makes the explanation sparse,
  which is what lifts the fidelity/stability numbers.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..xm_layers.revin import RevIN
except ImportError:  # allow running this file directly
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from xm_layers.revin import RevIN


# --------------------------------------------------------------------------- #
# Multi-scale temporal block
# --------------------------------------------------------------------------- #
class MultiScaleTemporalBlock(nn.Module):
    """Parallel dilated causal-ish convolutions at three temporal scales.

    With hourly data and kernel 3, two stacked layers give receptive
    fields of 5 h (local), 17 h (sub-daily) and 49 h (multi-day).
    Branches are combined with a learned per-timestep gate, not a
    concatenation — the gate weights are themselves a figure in the paper
    (contribution of each scale).
    """

    def __init__(self, d_out: int = 32, dilations: Tuple[int, ...] = (1, 4, 12),
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dilations = dilations
        self.n_branches = len(dilations)
        self.branches = nn.ModuleList()
        for d in dilations:
            pad = ((kernel_size - 1) * d) // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(1, d_out, kernel_size, padding=pad, dilation=d),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(d_out, d_out, kernel_size, padding=pad, dilation=d),
                    nn.GELU(),
                )
            )
        self.gate = nn.Conv1d(self.n_branches * d_out, self.n_branches, kernel_size=1)
        self.d_out = d_out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (BN, 1, L) -> (BN, d_out, L), gates (BN, n_branches, L)"""
        outs = [br(x) for br in self.branches]                    # each (BN, d_out, L)
        L = x.shape[-1]
        outs = [o[..., :L] for o in outs]                         # guard odd padding
        cat = torch.cat(outs, dim=1)                              # (BN, n*d_out, L)
        gates = torch.softmax(self.gate(cat), dim=1)              # (BN, n, L)
        stacked = torch.stack(outs, dim=1)                        # (BN, n, d_out, L)
        fused = (stacked * gates.unsqueeze(2)).sum(dim=1)         # (BN, d_out, L)
        return fused, gates


# --------------------------------------------------------------------------- #
# Encoder layer that returns its attention map (needed for rollout)
# --------------------------------------------------------------------------- #
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
        x = x + self.dropout(a)
        x = x + self.ff(self.norm2(x))
        return x, w                                               # w: (BN, P, P)


def attention_rollout(attns: List[torch.Tensor]) -> torch.Tensor:
    """attns: list of (BN, P, P) -> patch importance (BN, P).

    Abnar & Zuidema 2020: add the residual connection as an identity
    matrix, renormalize, multiply across layers. Column mass = how much
    each input patch is attended to overall.
    """
    device = attns[0].device
    P = attns[0].shape[-1]
    eye = torch.eye(P, device=device).unsqueeze(0)
    roll = None
    for a in attns:
        a = 0.5 * a + 0.5 * eye
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        roll = a if roll is None else torch.bmm(a, roll)
    imp = roll.mean(dim=1)                                        # (BN, P)
    return imp / imp.sum(dim=-1, keepdim=True).clamp_min(1e-9)


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class XAIMeteoFormer(nn.Module):
    def __init__(
        self,
        n_channels: int,
        target_idx: List[int],
        seq_len: int = 96,
        pred_len: int = 24,
        patch_len: int = 16,
        stride: int = 8,
        d_ms: int = 32,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.2,
        dilations: Tuple[int, ...] = (1, 4, 12),
        use_revin: bool = True,
        use_multiscale: bool = True,
        use_var_attn: bool = True,
        use_temp_attn: bool = True,
        use_fusion: bool = True,
        n_cls_outputs: int = 1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.register_buffer("target_idx", torch.tensor(target_idx, dtype=torch.long))
        self.n_targets = len(target_idx)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_cls_outputs = n_cls_outputs

        # ablation switches
        self.use_revin = use_revin
        self.use_multiscale = use_multiscale
        self.use_var_attn = use_var_attn
        self.use_temp_attn = use_temp_attn
        self.use_fusion = use_fusion

        self.num_patches = (seq_len - patch_len) // stride + 1
        d_ff = d_ff or 4 * d_model

        self.revin = RevIN(n_channels) if use_revin else None
        self.ms = MultiScaleTemporalBlock(d_ms, dilations, dropout=dropout) \
            if use_multiscale else None

        in_dim = (d_ms if use_multiscale else 1) * patch_len
        self.patch_proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        self.var_emb = nn.Embedding(n_channels, d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        # interpretable temporal pooling (single learned query)
        self.temporal_query = nn.Parameter(torch.randn(d_model) * 0.02)
        # interpretable variable attention (one query per target)
        self.variable_query = nn.Parameter(torch.randn(self.n_targets, d_model) * 0.02)

        self.fusion_gate = nn.Linear(2 * d_model, d_model) if use_fusion else None

        self.reg_w = nn.Parameter(torch.randn(self.n_targets, d_model, pred_len) * 0.02)
        self.reg_b = nn.Parameter(torch.zeros(self.n_targets, pred_len))
        self.cls_head = nn.Linear(d_model, pred_len * n_cls_outputs)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor,
                return_explanations: bool = False
                ) -> Dict[str, torch.Tensor]:
        """x: (B, L, N)

        Returns dict with:
            y_pred : (B, pred_len, n_targets)   original units
            logits : (B, pred_len, n_cls_outputs)
            var_entropy : scalar
        and, if return_explanations, also var_attn / temp_attn / rollout /
        scale_gates.
        """
        B, L, N = x.shape
        assert L == self.seq_len, f"expected seq_len={self.seq_len}, got {L}"
        assert N == self.n_channels, f"expected {self.n_channels} channels, got {N}"

        if self.revin is not None:
            x = self.revin.normalize(x)

        z = x.permute(0, 2, 1).reshape(B * N, 1, L)               # (BN, 1, L)

        scale_gates = None
        if self.ms is not None:
            z, scale_gates = self.ms(z)                            # (BN, d_ms, L)

        # patching
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # (BN, C, num_patches, patch_len)
        z = z.permute(0, 2, 1, 3).reshape(B * N, self.num_patches, -1)
        z = self.patch_proj(z)                                     # (BN, P, d_model)

        var_ids = torch.arange(N, device=x.device).repeat(B)       # matches b*N+n order
        z = z + self.pos_emb + self.var_emb(var_ids).unsqueeze(1)
        z = self.emb_drop(z)

        attns: List[torch.Tensor] = []
        for layer in self.layers:
            z, w = layer(z)
            attns.append(w)
        z = self.norm(z)                                           # (BN, P, d_model)

        # ---- temporal attention (interpretable pooling) ---------------- #
        if self.use_temp_attn:
            scores = (z @ self.temporal_query) / math.sqrt(self.d_model)  # (BN, P)
            alpha = torch.softmax(scores, dim=-1)
            h_t = (alpha.unsqueeze(-1) * z).sum(dim=1)             # (BN, d_model)
        else:
            alpha = torch.full((B * N, self.num_patches),
                               1.0 / self.num_patches, device=x.device)
            h_t = z.mean(dim=1)

        h_t = h_t.view(B, N, self.d_model)                         # (B, N, d_model)

        # ---- variable attention ---------------------------------------- #
        if self.use_var_attn:
            vs = torch.einsum("bnd,td->btn", h_t, self.variable_query)
            vs = vs / math.sqrt(self.d_model)
            beta = torch.softmax(vs, dim=-1)                       # (B, n_targets, N)
        else:
            beta = torch.full((B, self.n_targets, N), 1.0 / N, device=x.device)
        h_v = torch.einsum("btn,bnd->btd", beta, h_t)              # (B, n_targets, d)

        # ---- adaptive fusion ------------------------------------------- #
        h_self = h_t[:, self.target_idx, :]                        # (B, n_targets, d)
        if self.fusion_gate is not None:
            g = torch.sigmoid(self.fusion_gate(torch.cat([h_v, h_self], dim=-1)))
            h = g * h_self + (1.0 - g) * h_v
        else:
            h = h_self + h_v

        # ---- heads ------------------------------------------------------ #
        y = torch.einsum("btd,tdh->bth", h, self.reg_w) + self.reg_b
        y = y.permute(0, 2, 1)                                     # (B, pred_len, n_t)
        if self.revin is not None:
            y = self.revin.denormalize(y, self.target_idx)

        logits = self.cls_head(h.mean(dim=1))
        logits = logits.view(B, self.pred_len, self.n_cls_outputs)

        eps = 1e-9
        var_entropy = -(beta * (beta + eps).log()).sum(-1).mean()

        out = {"y_pred": y, "logits": logits, "var_entropy": var_entropy}
        if return_explanations:
            out["var_attn"] = beta                                 # (B, n_targets, N)
            out["temp_attn"] = alpha.view(B, N, self.num_patches)
            out["rollout"] = attention_rollout(attns).view(B, N, self.num_patches)
            if scale_gates is not None:
                out["scale_gates"] = scale_gates.view(B, N, -1, L).mean(-1)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, N, H = 8, 96, 21, 24
    model = XAIMeteoFormer(n_channels=N, target_idx=[1, 4, 0, 11],
                           seq_len=L, pred_len=H)
    x = torch.randn(B, L, N)
    out = model(x, return_explanations=True)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert out["y_pred"].shape == (B, H, 4), out["y_pred"].shape
    assert out["logits"].shape == (B, H, 1), out["logits"].shape
    assert out["var_attn"].shape == (B, 4, N), out["var_attn"].shape
    assert out["temp_attn"].shape == (B, N, model.num_patches)
    assert out["rollout"].shape == (B, N, model.num_patches)
    assert torch.allclose(out["var_attn"].sum(-1), torch.ones(B, 4), atol=1e-4)
    out["y_pred"].mean().backward()
    print(f"model ok | params={n_par/1e6:.2f}M | patches={model.num_patches}")
    for k, v in out.items():
        print(" ", k, tuple(v.shape) if v.dim() else float(v.detach()))
