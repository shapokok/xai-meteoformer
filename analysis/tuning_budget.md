# Tuning budget (Reviewer 2)

Reconstructed from the code and the recorded runs. No training.

## Direct answer

**There was no hyperparameter search. For any model.** Every one of the 138
recorded runs used the same architecture-level hyperparameters, and the only
value that ever varied is `lambda_ent`, which is an ablation (`no_entropy`), not
a search. The number of trials per model is **1**.

Evidence:

- No search framework appears anywhere in the repo — no Optuna, Ray Tune,
  hyperopt, or grid/random search code. The only occurrences of the word
  "sweep" in `src/` are comments about sweeping *seeds*.
- Across all 138 runs in [analysis/results_clean.csv](analysis/results_clean.csv)
  the distinct values are: `seq_len` {96}, `pred_len` {24}, `d_model` {256},
  `n_layers` {2}, `lr` {5e-4}, `lambda_ent` {0.0, 0.01}.
- The Kaggle notebook loops over `MODELS × SEEDS` only, passing no
  hyperparameter overrides.

## Per model

| Model | Search space explored | Trials | Configuration used | Source of the defaults |
|---|---|---|---|---|
| XAI-MeteoFormer | none | 1 | `d_model`=256, `n_heads`=8, `n_layers`=2, `patch_len`=16, `stride`=8, `dropout`=0.2, `lr`=5e-4, `batch`=64, AdamW + OneCycle, `lambda_cls`=0.2, `lambda_ent`=0.01 | our own `argparse` defaults, [src/train.py:352-390](src/train.py#L352-L390) |
| DLinear | none | 1 | shared config | TSLib defaults via `make_configs` |
| PatchTST | none | 1 | shared config, `patch_len`=16, `stride`=8 | TSLib defaults + our patch settings |
| iTransformer | none | 1 | shared config | TSLib defaults |
| Transformer | none | 1 | shared config, `e_layers`=2, `d_layers`=1, `factor`=3 | TSLib defaults |
| Informer | none | 1 | shared config, `distil`=True | TSLib defaults |
| Autoformer | none | 1 | shared config, `moving_avg`=25 | TSLib defaults |
| Crossformer | none | 1 | **`d_model`=128, `d_ff`=128**, `seg_len`=12 | `MODEL_OVERRIDES`, sized for 8 GB VRAM |
| TimesNet | none | 1 | **`d_model`=32, `d_ff`=32**, `e_layers`=2, `top_k`=3 | `MODEL_OVERRIDES`, sized for 8 GB VRAM |
| TFT | none | 1 | **`d_model`=128, `n_heads`=4** | `MODEL_OVERRIDES`, sized for 8 GB VRAM |
| LSTM | none | 1 | `hidden`=256, `layers`=2, `dropout`=0.2 | our own defaults |

Shared config for every model: `seq_len`=96, `pred_len`=24, `d_model`=256 unless
overridden, `d_ff`=4·`d_model`, `n_layers`=2, `dropout`=0.2, `lr`=5e-4,
`batch_size`=64, AdamW (`weight_decay`=1e-4), OneCycleLR (`pct_start`=0.3),
max 40 epochs, early stopping on validation loss with `patience`=6, gradient
clipping at 1.0, Huber loss (`delta`=1.0) on globally standardised targets.

## Runs and wall-clock

All times from the repaired `train_time_s`
([analysis/results_hygiene.md](analysis/results_hygiene.md) — the raw CSVs'
timing column is corrupted). Kaggle T4.

| Model | Runs | Mean min/run (Beijing) | Mean min/run (Jena) | Mean epochs |
|---|---|---|---|---|
| XAI-MeteoFormer | 38 | 5.1 | 10.6 | 19.6–20.9 |
| Crossformer | 10 | 12.3 | 22.2 | 13.0 |
| Autoformer | 10 | 5.9 | 23.4 | 12.9 |
| PatchTST | 10 | 6.0 | 20.3 | 16.7 |
| TFT | 10 | 7.1 | 16.1 | 10.2 |
| TimesNet | 10 | 6.9 | 15.7 | 10.4 |
| Informer | 10 | 2.5 | 5.8 | 12.6 |
| Transformer | 10 | 1.8 | 4.1 | 9.2 |
| iTransformer | 10 | 0.7 | 1.6 | 11.5 |
| LSTM | 10 | 0.6 | 1.1 | 10.5 |
| DLinear | 10 | 0.7 | 1.0 | 18.6 |
| **total** | **138** | | | **18.8 GPU-h** |

## How to present this, and the honest caveat

The fair framing is: *no model received any tuning; every model was trained
under an identical protocol with library-default or capacity-matched settings,
so the comparison is a controlled one rather than a tuned-vs-tuned one.* That is
a legitimate and increasingly common protocol, and it is stronger than a
comparison where only the proposed model was tuned.

The caveat must be stated in the same breath, because a reviewer will raise it:
a no-tuning protocol can disadvantage baselines whose published configurations
differ from ours. Three specifics worth naming:

1. **Crossformer, TimesNet and TFT ran at reduced width** (128/32/128 instead of
   256) to fit 8 GB of VRAM, while our model ran at full 256. Crossformer is the
   closest competitor, so this cuts against us being accused of favouring
   ourselves — but it also means Crossformer's numbers are not its best possible
   numbers, and its RMSE advantage on Jena was obtained *while under-provisioned*.
2. **PatchTST was given our `patch_len`/`stride`**, not the values from its own
   paper.
3. `lr`=5e-4 with OneCycle is our choice, applied to every model including those
   whose reference implementations use a different schedule.

None of this requires new runs to state. It requires one honest paragraph in the
experimental-setup section.
