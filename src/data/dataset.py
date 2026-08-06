"""Sliding-window dataset with a chronological split.

Two rules here are what reviewers check first, so they are enforced
rather than left to convention:

1. The split is chronological 70/10/20, never shuffled.
2. The scaler is fitted on the training segment only. Fitting on the
   full series leaks future statistics into the test set.

Windows are also clipped so that no window crosses a split boundary.

The classification label is frost (T <= 0 degC) at each of the 24
forecast steps, computed from the RAW target values before scaling.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MeteoWindowDataset(Dataset):
    SPLITS = {"train": 0, "val": 1, "test": 2}

    def __init__(
        self,
        processed_dir: str,
        name: str,
        split: str = "train",
        seq_len: int = 96,
        pred_len: int = 24,
        frost_threshold: float = 0.0,
        missing_rate: float = 0.0,
        scaler: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        seed: int = 0,
    ):
        assert split in self.SPLITS
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split = split
        self.missing_rate = missing_rate
        self.rng = np.random.default_rng(seed)

        X = np.load(os.path.join(processed_dir, f"{name}_X.npy"))
        with open(os.path.join(processed_dir, f"{name}_meta.json")) as f:
            self.meta = json.load(f)
        self.target_idx: List[int] = self.meta["target_idx"]
        self.target_names: List[str] = self.meta["target_names"]
        self.columns: List[str] = self.meta["columns"]
        self.n_channels = X.shape[1]

        T = X.shape[0]
        n_tr, n_va = int(T * 0.7), int(T * 0.1)
        bounds = {"train": (0, n_tr),
                  "val": (n_tr, n_tr + n_va),
                  "test": (n_tr + n_va, T)}

        if scaler is None:
            tr = X[: n_tr]
            self.mu = tr.mean(axis=0)
            self.sigma = tr.std(axis=0)
            self.sigma[self.sigma < 1e-8] = 1.0
        else:
            self.mu, self.sigma = scaler

        lo, hi = bounds[split]
        self.raw = X[lo:hi]
        self.scaled = ((self.raw - self.mu) / self.sigma).astype(np.float32)

        n_valid = len(self.raw) - seq_len - pred_len + 1
        if n_valid <= 0:
            raise ValueError(
                f"split '{split}' has {len(self.raw)} rows, too short for "
                f"seq_len={seq_len} + pred_len={pred_len}"
            )
        self.n_valid = n_valid

        tcol = self.target_idx[self.target_names.index("T")]
        self.frost = (X[lo:hi, tcol] <= frost_threshold).astype(np.float32)
        self.frost_rate = float(self.frost.mean())

    def scaler_params(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.mu, self.sigma

    def __len__(self) -> int:
        return self.n_valid

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        s, e = i, i + self.seq_len
        f = e + self.pred_len

        x = self.scaled[s:e].copy()                         # (L, N)
        if self.missing_rate > 0:
            # Robustness experiment: drop observations at random and fill
            # forward, which is what an operational gap looks like.
            mask = self.rng.random(x.shape) < self.missing_rate
            x[mask] = np.nan
            idx = np.where(~np.isnan(x[:, :1]).squeeze(-1))[0]
            for c in range(x.shape[1]):
                col = x[:, c]
                bad = np.isnan(col)
                if bad.all():
                    col[:] = 0.0
                elif bad.any():
                    good = np.where(~bad)[0]
                    col[bad] = np.interp(np.where(bad)[0], good, col[good])
            del idx

        y_scaled = self.scaled[e:f][:, self.target_idx]      # (H, n_targets)
        y_raw = self.raw[e:f][:, self.target_idx]
        cls = self.frost[e:f][:, None]                       # (H, 1)

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "y": torch.from_numpy(np.ascontiguousarray(y_scaled)),
            "y_raw": torch.from_numpy(np.ascontiguousarray(y_raw)),
            "cls": torch.from_numpy(np.ascontiguousarray(cls)),
        }


def build_splits(processed_dir: str, name: str, **kw):
    """Train split defines the scaler; val/test reuse it."""
    train = MeteoWindowDataset(processed_dir, name, split="train", **kw)
    sc = train.scaler_params()
    val = MeteoWindowDataset(processed_dir, name, split="val", scaler=sc, **kw)
    test = MeteoWindowDataset(processed_dir, name, split="test", scaler=sc, **kw)
    return train, val, test


if __name__ == "__main__":
    # Self-test on a synthetic dataset so it runs before real data exists.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        T, N = 5000, 12
        t = np.arange(T)
        X = np.stack([
            10 + 8 * np.sin(2 * np.pi * t / 24) + np.random.randn(T),   # T
            60 + 15 * np.cos(2 * np.pi * t / 24) + np.random.randn(T),  # RH
            1013 + np.random.randn(T) * 3,                              # P
            np.abs(2 + np.random.randn(T)),                             # WS
        ] + [np.random.randn(T) for _ in range(N - 4)], axis=1).astype(np.float32)
        np.save(os.path.join(d, "toy_X.npy"), X)
        with open(os.path.join(d, "toy_meta.json"), "w") as f:
            json.dump({"name": "toy",
                       "columns": ["T", "RH", "P", "WS"] + [f"c{i}" for i in range(N - 4)],
                       "target_idx": [0, 1, 2, 3],
                       "target_names": ["T", "RH", "P", "WS"]}, f)

        tr, va, te = build_splits(d, "toy", seq_len=96, pred_len=24)
        b = tr[0]
        assert b["x"].shape == (96, N), b["x"].shape
        assert b["y"].shape == (24, 4) and b["y_raw"].shape == (24, 4)
        assert b["cls"].shape == (24, 1)
        assert np.allclose(tr.mu, va.mu) and np.allclose(tr.sigma, te.sigma), \
            "val/test must reuse the train scaler"
        assert len(tr) + len(va) + len(te) < T, "windows overlap a split boundary"
        noisy = MeteoWindowDataset(d, "toy", split="test", missing_rate=0.2,
                                   scaler=tr.scaler_params())
        assert torch.isfinite(noisy[0]["x"]).all(), "gap filling left NaNs"
        print(f"dataset ok | train={len(tr)} val={len(va)} test={len(te)} "
              f"| frost_rate={tr.frost_rate:.3f}")
