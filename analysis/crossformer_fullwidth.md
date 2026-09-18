# Crossformer at full width (P0-1)

Produced by [analysis/crossformer_fullwidth.py](analysis/crossformer_fullwidth.py). Jena, 5 seeds, mean ± sd. The normalization configuration is unchanged (none, as in the published runs).

`MODEL_OVERRIDES` capped Crossformer at `d_model=d_ff=128` with a comment about 8 GB of VRAM, but training ran on a 16 GB T4, so the cap was inherited rather than required. Measured peak memory at 256x1024, batch 64 is in `analysis/crossformer_probe.json`.

## Accuracy by capacity

| Configuration | params | n seeds | MAE | RMSE | R² |
|---|---|---|---|---|---|
| Crossformer 128x128 | 1,695,908 | 5 | 3.107 ± 0.018 | 5.166 ± 0.021 | 0.678 ± 0.004 |
| Crossformer 256x512 | — | 0 | _not run yet_ | — | — |
| Crossformer 256x1024 | — | 0 | _not run yet_ | — | — |
| MeteoFormer (ours) | 1,892,475 | 5 | 3.025 ± 0.031 | 5.267 ± 0.046 | 0.682 ± 0.005 |

**Missing: Crossformer 256x512, Crossformer 256x1024** — run cells 6/7 of `notebooks/kaggle_resubmit.ipynb`, download, then re-run `analysis/repair_results_csv.py` and this script.

## Diebold–Mariano against the proposed model

| Configuration | Δ L1 | p | Δ L2 | p |
|---|---|---|---|---|
| Crossformer 128x128 | -0.0824* | 9.93e-12 | +1.0513* | 1.63e-07 |
| Crossformer 256x512 | _not run yet_ | | | |
| Crossformer 256x1024 | _not run yet_ | | | |

Negative Δ favours the proposed model. `*` = p<0.05 after Holm within each column. L1 = absolute loss, L2 = squared loss; the per-window loss is averaged over seeds, not the predictions.

## How to read this

If Crossformer improves with width, the published comparison under-provisioned it and the paper must say so — reporting the capped configuration without that caveat would not survive review. If it does not improve, the cap was harmless and the existing numbers stand, which is worth one sentence in the setup section.

Either way the **parameter column stays in the table**: at 256 wide Crossformer is 4–6× the proposed model, so a win at that size is a statement about capacity, not about architecture.
