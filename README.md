# XAI-MeteoFormer

Reference implementation for the paper *XAI-MeteoFormer: An Explainable
Multi-Scale Dual-Attention Network for Meteorological Time-Series
Forecasting*.

```
src/
  layers/revin.py            reversible instance norm (target-subset aware)
  models/xai_meteoformer.py  the model
  data/prepare.py            Jena + Beijing -> common npy format
  data/dataset.py            windowing, chronological split, train-only scaler
  train.py                   training loop with resume + metric logging
```

## 0. Smoke test first (no GPU needed)

Run these three before touching Kaggle. Each is a few seconds on CPU and
catches every shape bug in the pipeline.

```bash
python src/layers/revin.py
python src/models/xai_meteoformer.py
python src/data/dataset.py
```

Expected: `revin ok`, `model ok | params=…M | patches=11`, `dataset ok | …`.

## 1. Data

```bash
python src/data/prepare.py --dataset jena    --out data/processed
python src/data/prepare.py --dataset beijing --out data/processed --raw data/raw
```

Jena downloads itself. Beijing (UCI PRSA, 12 stations, 2013–2017) has to
be present in `--raw`; on Kaggle add the public dataset as a notebook
input instead of downloading it.

Both end up with the same four targets — `T`, `RH`, `P`, `WS` — at hourly
resolution. Beijing has no humidity column, so RH is derived from
temperature and dew point via the Magnus formula.

Upload `data/processed/` as a Kaggle Dataset once. Re-running
preprocessing inside every training notebook burns GPU quota for nothing.

## 2. Training

```bash
# one run
python src/train.py --dataset jena --seed 0

# the five seeds the paper needs
python src/train.py --dataset jena --seeds 0 1 2 3 4

# ablation table
for a in full no_revin no_multiscale no_var_attn no_temp_attn no_fusion no_entropy; do
  python src/train.py --dataset jena --ablation $a --seeds 0 1 2
done

# missing-data robustness
for r in 0.05 0.10 0.20; do
  python src/train.py --dataset jena --missing_rate $r --seeds 0 1 2
done
```

Every finished run appends a row to `outputs/results.csv`. Re-running the
same command skips whatever is already there, so a 12-hour Kaggle
timeout costs one run, not the batch. `--force` overrides.

## 3. Compute split

| Where | What | Why |
|---|---|---|
| Kaggle (P100 / 2×T4) | XAI-MeteoFormer, DLinear, LSTM, Transformer, Informer, Autoformer, PatchTST, iTransformer; ablation; XAI | 30 GPU-h/week is enough for these |
| Windows laptop, RTX 3070 8 GB | TimesNet, TFT, Crossformer | slowest models, ~1/3 of total budget — keep them off the quota |
| MacBook M5 Pro | writing, figures from saved CSVs | MPS silently falls back to CPU on several TSLib ops |

### Batch sizes for 8 GB VRAM

| Model | batch | notes |
|---|---|---|
| XAI-MeteoFormer | 64 | ~0.9 GB at L=96 |
| DLinear / LSTM / Transformer | 64 | |
| Informer / Autoformer | 32 | |
| PatchTST / iTransformer | 64 | |
| **TFT** | **32**, hidden 128 | variable selection nets are memory-hungry |
| **TimesNet** | **16**, d_model 32, d_ff 32, top_k 3 | the 2D reshape is the constraint; drop to 8 if it OOMs |
| Crossformer | 16 | |

Always run with `--amp`. On the 3070 that is roughly a 1.7× speedup and
halves memory; on P100 the gain is smaller but never negative.

## 4. Kaggle rules that actually matter

1. **Debug with GPU off.** CPU sessions do not consume quota. Turn the
   GPU on only once the smoke tests pass.
2. **Use Save & Run All (commit), not the interactive session.** The
   interactive one dies when you close the laptop. You can hold one
   interactive plus two background sessions at the same time — that is
   three runs in parallel.
3. **Chain notebooks:** `00_data_prep` → output becomes the input dataset
   of `01_train` → its checkpoints become the input of `02_xai` →
   `03_figures`. Nothing is recomputed.
4. **Enable internet** (needs phone verification) so the notebook can
   `git clone` this repo instead of you pasting cells.
5. Keep `outputs/results.csv` and `outputs/predictions/` in the notebook
   output — the XAI and figure stages read from them.

## 5. Protocol (fixed — do not vary between models)

* Input length 96 h, forecast horizon 24 h, hourly resolution.
* Errors at h = 1, 6, 12, 24 come from the same trained model. The
  horizon axis costs no extra runs.
* Chronological split 70/10/20, scaler fitted on train only, windows
  never cross a split boundary.
* Huber loss, AdamW, OneCycle schedule, early stopping on validation MSE,
  patience 6.
* 5 seeds per configuration, reported as mean ± std.
* MAPE only for `P` and `RH`. Temperature in Celsius crosses zero and
  makes MAPE meaningless — sMAPE is reported for all four targets.
* **Baselines get the same hyperparameter search budget as the proposed
  model.** Tuning only the proposed model and leaving baselines on
  defaults is the first thing a reviewer will ask about.

## 6. Loss

```
Huber(y, ŷ)  +  0.2 · BCE(frost logits)  +  0.01 · H(variable attention)
```

The entropy term makes the variable attention sparse. It is not
cosmetic: sparse attention is what lifts the fidelity and stability
numbers that carry the paper's main contribution, and it is the concrete
difference from TFT.
