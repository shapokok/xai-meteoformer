# Crossformer at full width (P0-1)

Produced by [analysis/crossformer_fullwidth.py](analysis/crossformer_fullwidth.py). Jena, 5 seeds, mean ± sd. The normalization configuration is unchanged (none, as in the published runs).

`MODEL_OVERRIDES` capped Crossformer at `d_model=d_ff=128` with a comment about 8 GB of VRAM, but training ran on a 16 GB T4, so the cap was inherited rather than required. Measured peak memory at 256x1024, batch 64 is in `analysis/crossformer_probe.json`.

**Training setup of the 256-wide runs:** batch size **64** for both widths (the `CROSSFORMER_BATCH` default; batch is in neither the results row nor the checkpoint tag), Tesla T4, one process per GPU. Memory probe at 256x1024, one epoch: peak 815 MiB at batch 64 and 1464 MiB at batch 128, so memory did not constrain the choice.

## Accuracy by capacity

| Configuration | params | n seeds | MAE | RMSE | R² |
|---|---|---|---|---|---|
| Crossformer 128x128 | 1,695,908 | 5 | 3.107 ± 0.018 | 5.166 ± 0.021 | 0.678 ± 0.004 |
| Crossformer 256x512 | 7,981,860 | 5 | 3.184 ± 0.073 | 5.241 ± 0.063 | 0.675 ± 0.004 |
| Crossformer 256x1024 | 10,608,420 | 5 | 3.175 ± 0.042 | 5.237 ± 0.039 | 0.672 ± 0.005 |
| MeteoFormer (ours) | 1,892,475 | 5 | 3.025 ± 0.031 | 5.267 ± 0.046 | 0.682 ± 0.005 |

## Diebold–Mariano against the proposed model

| Configuration | Δ L1 | p | Δ L2 | p |
|---|---|---|---|---|
| Crossformer 128x128 | -0.0824* | 9.93e-12 | +1.0513* | 4.9e-07 |
| Crossformer 256x512 | -0.1595* | <1e-15 | +0.2659 | 0.273 |
| Crossformer 256x1024 | -0.1498* | <1e-15 | +0.3161 | 0.273 |

Negative Δ favours the proposed model. `*` = p<0.05 after Holm within each column. L1 = absolute loss, L2 = squared loss; the per-window loss is averaged over seeds, not the predictions.

## Result

- Crossformer 256x512: MAE 3.184 vs 3.107 at 128x128 (+0.077); vs ours 3.025 (-0.159).
- Crossformer 256x1024: MAE 3.175 vs 3.107 at 128x128 (+0.067); vs ours 3.025 (-0.150).

## How to read this

If Crossformer improves with width, the published comparison under-provisioned it and the paper must say so — reporting the capped configuration without that caveat would not survive review. If it does not improve, the cap was harmless and the existing numbers stand, which is worth one sentence in the setup section.

Either way the **parameter column stays in the table**: at 256 wide Crossformer is 4–6× the proposed model, so a win at that size is a statement about capacity, not about architecture.
