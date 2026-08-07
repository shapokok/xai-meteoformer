"""Baselines behind the same interface as XAI-MeteoFormer.

Every baseline must see the SAME inputs, the SAME split and the SAME
training loop as the proposed model, otherwise the comparison table is
worthless. This wrapper enforces that: TSLib models are constructed from
one config object and their output is reshaped into the dict our
train.py expects.

Setup on Kaggle / locally:

    git clone https://github.com/thuml/Time-Series-Library.git
    export TSLIB_PATH=$PWD/Time-Series-Library

Classification for baselines: they have no event head, so the score is
the negative predicted temperature. That is not a handicap — it is the
natural "threshold the regression" baseline, and gives a valid ranking
for ROC/PR. The comparison against the dedicated head is a table row in
its own right.
"""

import os
import sys
from types import SimpleNamespace
from typing import List, Optional

import torch
import torch.nn as nn

try:
    from ..xm_layers.revin import RevIN
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from xm_layers.revin import RevIN


# name -> (tslib module, needs decoder input)
TSLIB_MODELS = {
    "DLinear": ("DLinear", False),
    "PatchTST": ("PatchTST", False),
    "iTransformer": ("iTransformer", False),
    "TimesNet": ("TimesNet", False),
    "Crossformer": ("Crossformer", False),
    "TimeMixer": ("TimeMixer", False),
    "Transformer": ("Transformer", True),
    "Informer": ("Informer", True),
    "Autoformer": ("Autoformer", True),
    "FEDformer": ("FEDformer", True),
    "TFT": ("TemporalFusionTransformer", True),
}
LOCAL_MODELS = ["LSTM", "GRU"]
AVAILABLE = ["XAI-MeteoFormer"] + LOCAL_MODELS + list(TSLIB_MODELS)

# Per-model overrides. Sized for 8 GB VRAM; see README.
MODEL_OVERRIDES = {
    "TimesNet": dict(d_model=32, d_ff=32, e_layers=2, top_k=3),
    "Crossformer": dict(d_model=128, d_ff=128, seg_len=12),
    "TFT": dict(d_model=128, n_heads=4),
    # TimeMixer indexes per-scale lists; with 0 down-sampling layers the
    # list is empty and its constructor raises IndexError.
    "TimeMixer": dict(d_model=128, d_ff=128, down_sampling_layers=1,
                      down_sampling_window=2, down_sampling_method="avg"),
    "DLinear": dict(),
}


def _neutralize_reformer_import(tslib_path: str) -> None:
    """Make TSLib importable without reformer_pytorch.

    layers/SelfAttention_Family.py imports reformer_pytorch at module level,
    which every attention-based model then pulls in. reformer_pytorch drags a
    chain of its own (local_attention -> hyper_connections -> ...), and
    installing it with dependencies can also replace the environment's torch
    with a wheel built for different GPU architectures.

    None of our baselines use Reformer, so the import is simply made
    optional. LSHSelfAttention is referenced only inside ReformerLayer's
    __init__, so leaving it as None is harmless unless that layer is built.
    """
    f = os.path.join(tslib_path, "layers", "SelfAttention_Family.py")
    if not os.path.exists(f):
        return
    with open(f) as fh:
        txt = fh.read()
    needle = "from reformer_pytorch import LSHSelfAttention"
    if needle not in txt or "XAIMF_OPTIONAL_REFORMER" in txt:
        return
    with open(f, "w") as fh:
        fh.write(txt.replace(needle,
            "try:  # XAIMF_OPTIONAL_REFORMER\n"
            "    from reformer_pytorch import LSHSelfAttention\n"
            "except ImportError:\n"
            "    LSHSelfAttention = None"))
    print("  [tslib] made reformer_pytorch import optional")


def _register_tft_layout(mod, cfg, n_channels: int) -> bool:
    """TSLib's TFT reads its covariate layout from `datatype_dict`, keyed by
    dataset name. Every built-in entry hard-codes a channel count (ETTh1 is
    7), so reusing one would silently feed TFT only the first 7 of our
    channels and cripple the closest competitor to our model. Register an
    entry with all channels observed and no static covariates instead.
    """
    dd = getattr(mod, "datatype_dict", None)
    TypePos = getattr(mod, "TypePos", None)
    if dd is None or TypePos is None:
        return False
    key = "xaimeteoformer"
    dd[key] = TypePos([], list(range(n_channels)))
    cfg.data = key
    return True


def _instantiate(mod, cfg, name: str):
    """TSLib's TFT picks its covariate layout from a dict keyed by dataset
    name, and 'custom' is not one of the keys. Probe the known ones."""
    try:
        return mod.Model(cfg)
    except KeyError as e:
        keys = set()
        for k_, v in vars(mod).items():
            if k_ == "__builtins__":
                continue
            if isinstance(v, dict) and v and all(isinstance(k, str) for k in v):
                keys |= set(v)
        candidates = [k for k in ("weather", "ETTh1", "ETTm1", "electricity",
                                  "traffic", "exchange_rate") if k in keys]
        candidates += [k for k in sorted(keys) if k not in candidates]
        for cand in candidates:
            cfg.data = cand
            try:
                core = mod.Model(cfg)
                print(f"  [{name}] configs.data={cand!r} accepted "
                      f"(rejected {e}); available: {sorted(keys)}")
                return core
            except (KeyError, TypeError, ValueError):
                continue
        raise KeyError(
            f"{name}: no usable configs.data value. Tried {candidates}"
        ) from e


def make_configs(args, n_channels: int, **overrides) -> SimpleNamespace:
    """One config object broad enough for every TSLib long-term forecaster."""
    cfg = SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=args.seq_len,
        label_len=args.seq_len // 2,
        pred_len=args.pred_len,
        enc_in=n_channels, dec_in=n_channels, c_out=n_channels,
        d_model=args.d_model, n_heads=args.n_heads,
        e_layers=args.n_layers, d_layers=1,
        d_ff=4 * args.d_model,
        dropout=args.dropout,
        factor=3, moving_avg=25, distil=True,
        embed="timeF", freq="h", activation="gelu",
        output_attention=False,
        num_kernels=6, top_k=5,
        patch_len=args.patch_len, stride=args.stride,
        seg_len=12, win_size=2,
        channel_independence=1, use_norm=1,
        decomp_method="moving_avg",
        down_sampling_layers=0, down_sampling_window=1,
        down_sampling_method="avg",
        expand=2, d_conv=4,
        p_hidden_dims=[128, 128], p_hidden_layers=2,
        features="M", num_class=1, node_dim=10, dec_way="pmf",
        # TFT reads configs.data to decide its covariate layout
        data="custom", target="T", num_targets=1,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class SimpleRNN(nn.Module):
    """LSTM / GRU baseline. Gets RevIN too — the TSLib models all carry
    some internal normalization, so withholding it here would stack the
    deck against the classic baseline."""

    def __init__(self, kind: str, n_channels: int, n_targets: int,
                 pred_len: int, hidden: int = 256, layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        rnn = nn.LSTM if kind == "LSTM" else nn.GRU
        self.revin = RevIN(n_channels)
        self.rnn = rnn(n_channels, hidden, num_layers=layers,
                       batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, pred_len * n_targets)
        self.pred_len, self.n_targets = pred_len, n_targets

    def forward(self, x):
        x = self.revin.normalize(x)
        out, _ = self.rnn(x)
        y = self.head(out[:, -1])
        return y.view(x.size(0), self.pred_len, self.n_targets), self.revin


class BaselineWrapper(nn.Module):
    """Presents any baseline through the XAI-MeteoFormer output contract."""

    def __init__(self, core: nn.Module, name: str, target_idx: List[int],
                 seq_len: int, pred_len: int, needs_dec: bool,
                 n_time_feats: int = 4, temp_pos: int = 0,
                 is_rnn: bool = False):
        super().__init__()
        self.core = core
        self.name = name
        self.register_buffer("target_idx", torch.tensor(target_idx, dtype=torch.long))
        self.seq_len, self.pred_len = seq_len, pred_len
        self.label_len = seq_len // 2
        self.needs_dec = needs_dec
        self.n_time_feats = n_time_feats
        self.temp_pos = temp_pos
        self.is_rnn = is_rnn

    def forward(self, x: torch.Tensor, return_explanations: bool = False):
        B = x.size(0)
        if self.is_rnn:
            y, revin = self.core(x)
            y = revin.denormalize(y, self.target_idx)
        else:
            # The last four channels are hour_sin/cos, doy_sin/cos, which
            # is exactly what embed='timeF' with freq='h' expects.
            x_mark = x[:, :, -self.n_time_feats:]
            if self.needs_dec:
                zeros = torch.zeros(B, self.pred_len, x.size(-1), device=x.device)
                x_dec = torch.cat([x[:, -self.label_len:, :], zeros], dim=1)
                mark_zeros = torch.zeros(B, self.pred_len, self.n_time_feats,
                                         device=x.device)
                x_mark_dec = torch.cat([x_mark[:, -self.label_len:, :],
                                        mark_zeros], dim=1)
            else:
                x_dec, x_mark_dec = None, None
            out = self.core(x, x_mark, x_dec, x_mark_dec)
            if isinstance(out, (tuple, list)):
                out = out[0]
            y = out[:, -self.pred_len:, :]              # (B, H, N)
            y = y[:, :, self.target_idx]                # (B, H, n_targets)

        # threshold-the-regression score for the event task
        logits = -y[:, :, self.temp_pos:self.temp_pos + 1]

        res = {"y_pred": y, "logits": logits,
               "var_entropy": torch.zeros((), device=x.device)}
        if return_explanations:
            res["var_attn"] = None
            res["temp_attn"] = None
        return res


def build_baseline(name: str, args, n_channels: int,
                   target_idx: List[int], target_names: List[str],
                   tslib_path: Optional[str] = None) -> nn.Module:
    temp_pos = target_names.index("T") if "T" in target_names else 0

    if name in LOCAL_MODELS:
        core = SimpleRNN(name, n_channels, len(target_idx), args.pred_len,
                         hidden=args.d_model, layers=args.n_layers,
                         dropout=args.dropout)
        return BaselineWrapper(core, name, target_idx, args.seq_len,
                               args.pred_len, needs_dec=False,
                               temp_pos=temp_pos, is_rnn=True)

    if name not in TSLIB_MODELS:
        raise ValueError(f"unknown model '{name}'. Available: {AVAILABLE}")

    tslib_path = tslib_path or os.environ.get("TSLIB_PATH")
    if not tslib_path or not os.path.isdir(tslib_path):
        raise FileNotFoundError(
            "Time-Series-Library not found. Clone it and set TSLIB_PATH:\n"
            "  git clone https://github.com/thuml/Time-Series-Library.git\n"
            "  export TSLIB_PATH=$PWD/Time-Series-Library"
        )
    # Appended, not prepended: our own packages are named xm_layers /
    # xm_models precisely so TSLib's layers/ and models/ resolve to
    # TSLib. Prepending would also shadow our data/ package.
    _neutralize_reformer_import(tslib_path)
    if tslib_path not in sys.path:
        sys.path.append(tslib_path)

    module_name, needs_dec = TSLIB_MODELS[name]
    try:
        mod = __import__(f"models.{module_name}", fromlist=["Model"])
    except ImportError as e:
        avail = sorted(f[:-3] for f in os.listdir(os.path.join(tslib_path, "models"))
                       if f.endswith(".py") and not f.startswith("_"))
        missing = str(e).split("'")[1] if "'" in str(e) else ""
        if missing and not missing.startswith(("layers", "models", "utils")):
            raise ImportError(
                f"'{module_name}' needs a package that is not installed: "
                f"{missing}. Run:  pip install matplotlib einops"
            ) from e
        raise ImportError(
            f"'{module_name}' not in this TSLib checkout ({e}). Present: {avail}"
        ) from e

    cfg = make_configs(args, n_channels, **MODEL_OVERRIDES.get(name, {}))
    if module_name == "TemporalFusionTransformer":
        if _register_tft_layout(mod, cfg, n_channels):
            print(f"  [{name}] registered covariate layout with all "
                  f"{n_channels} channels observed")
    core = _instantiate(mod, cfg, name)
    return BaselineWrapper(core, name, target_idx, args.seq_len, args.pred_len,
                           needs_dec=needs_dec, temp_pos=temp_pos)


if __name__ == "__main__":
    from types import SimpleNamespace as NS

    args = NS(seq_len=96, pred_len=24, d_model=256, n_heads=8, n_layers=2,
              dropout=0.2, patch_len=16, stride=8)
    B, N = 4, 19
    x = torch.randn(B, args.seq_len, N)
    names = ["T", "RH", "P", "WS"]

    for m in LOCAL_MODELS:
        w = build_baseline(m, args, N, [0, 1, 2, 3], names)
        o = w(x)
        assert o["y_pred"].shape == (B, 24, 4), (m, o["y_pred"].shape)
        assert o["logits"].shape == (B, 24, 1)
        print(f"{m:14s} ok  params={sum(p.numel() for p in w.parameters())/1e6:.2f}M")

    if os.environ.get("TSLIB_PATH"):
        for m in TSLIB_MODELS:
            try:
                w = build_baseline(m, args, N, [0, 1, 2, 3], names)
                o = w(x)
                assert o["y_pred"].shape == (B, 24, 4), (m, o["y_pred"].shape)
                print(f"{m:14s} ok  "
                      f"params={sum(p.numel() for p in w.parameters())/1e6:.2f}M")
            except Exception as e:  # keep going, report the rest
                print(f"{m:14s} FAILED: {type(e).__name__}: {e}")
    else:
        print("\nTSLIB_PATH not set — skipped TSLib models. Set it and re-run "
              "to verify all baselines instantiate.")
