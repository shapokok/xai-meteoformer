# Reproducibility

Everything below is read off the code and the recorded runs, not recalled from
memory. File references are clickable.

## 1. Data

| | Jena | Beijing (Aotizhongxin) |
|---|---|---|
| Source | `jena_climate_2009_2016.csv` (TF datasets mirror, downloaded by [src/data/prepare.py](src/data/prepare.py)) | UCI *Beijing Multi-Site Air-Quality*, station Aotizhongxin |
| Native resolution | 10 min | 1 h |
| After resampling | 1 h | 1 h |
| Rows | 70 129 | 35 064 |
| Channels | 19 | 18 |
| Period | 2009-01-01 00:00 → 2017-01-01 00:00 | 2013-03-01 00:00 → 2017-02-28 23:00 |
| Targets | T, RH, P, WS (indices 0–3) | T, RH, P, WS (indices 0–3) |

## 2. Missing-value handling

Jena ([prepare_jena](src/data/prepare.py#L59)):

1. The wind columns `wv (m/s)` and `max. wv (m/s)` use **−9999 as a missing
   sentinel**; those entries are set to `0.0` before anything else.
2. Wind direction is converted to vector components **before** averaging
   (`wx = wv·cos θ`, `wy = wv·sin θ`), because the hourly mean of 359° and 1°
   would otherwise be 180°. `wd (deg)` is then dropped.
3. `df.resample("1h").mean()` — 10-minute records aggregated to hourly.
4. `df.interpolate(limit_direction="both")` — linear interpolation over the
   remaining gaps, extended to both ends so no NaN survives at the boundaries.

Beijing ([prepare_beijing](src/data/prepare.py#L163)):

1. Wind direction is categorical (`N`, `NNE`, …); mapped to degrees via `WD_MAP`
   and converted to `wx`/`wy` as above.
2. `df.apply(pd.to_numeric, errors="coerce")` turns any unparseable entry into
   NaN, then the same `interpolate(limit_direction="both")`.
3. **RH is not in the source** and is derived from temperature and dew point
   with the Magnus formula (Alduchov & Eskridge 1996 coefficients,
   `a=17.625`, `b=243.04`), clipped to [0, 100], so that Beijing carries the
   same four targets as Jena.

Both paths end with hard assertions that no NaN and no non-finite value survives
([prepare.py:245-246](src/data/prepare.py#L245-L246)).

## 3. Feature construction

- Four cyclic time features appended to every dataset:
  `hour_sin`, `hour_cos` (period 24 h) and `doy_sin`, `doy_cos` (period 365.25 d).
  These are the last four channels, which is what the TSLib baselines'
  `embed="timeF", freq="h"` expects.
- Column order is always **targets first**, so `target_idx` is `[0,1,2,3]` in
  every dataset.
- A reduced 11-channel "common" variant exists
  (`{T, RH, P, WS, Tdew, wx, wy, hour_sin, hour_cos, doy_sin, doy_cos}`) for
  cross-dataset transfer. **It was not used for any result in the paper** — all
  138 runs use the full channel set.

## 4. Splits — exact dates

Chronological 70/10/20, never shuffled. Windows are clipped so none crosses a
boundary. The scaler (per-channel mean/sd) is fitted on the **training segment
only** and reused by validation and test
([src/data/dataset.py](src/data/dataset.py)).

**Jena** (`seq_len`=96, `pred_len`=24):

| Split | Rows | Dates | Windows |
|---|---|---|---|
| train | [0 : 49090] | 2009-01-01 00:00 → 2014-08-08 09:00 | 48 971 |
| val | [49090 : 56102] | 2014-08-08 10:00 → 2015-05-27 13:00 | 6 893 |
| test | [56102 : 70129] | 2015-05-27 14:00 → 2017-01-01 00:00 | 13 908 |

**Beijing (Aotizhongxin)**:

| Split | Rows | Dates | Windows |
|---|---|---|---|
| train | [0 : 24544] | 2013-03-01 00:00 → 2015-12-18 15:00 | 24 425 |
| val | [24544 : 28050] | 2015-12-18 16:00 → 2016-05-12 17:00 | 3 387 |
| test | [28050 : 35064] | 2016-05-12 18:00 → 2017-02-28 23:00 | 6 895 |

The split depends only on the row count, not on the seed: the `seed` argument to
`MeteoWindowDataset` seeds only the RNG used for the synthetic-missingness
experiment, which is off (`missing_rate=0`) for every reported run. Consequently
the test set is byte-identical across all models and seeds, which is what makes
a single shared `predictions/{dataset}_true.npy` valid.

## 5. Training protocol

Identical for every model, proposed and baseline
([src/train.py](src/train.py)):

| | |
|---|---|
| Input window / horizon | 96 h → 24 h |
| Loss | `HuberLoss(delta=1.0)` on globally standardised targets, `+ 0.2·BCEWithLogits` (frost head), `+ 0.01·var_entropy` (proposed model only) |
| Class imbalance | `pos_weight = (1−r)/r`, `r` = frost rate of the training split |
| Optimiser | AdamW, `weight_decay`=1e-4 |
| LR schedule | OneCycleLR, `max_lr`=5e-4, `pct_start`=0.3 |
| Batch size | 64 |
| Epochs | max 40, early stopping on validation loss, `patience`=6 |
| Gradient clipping | 1.0 |
| Mixed precision | AMP on CUDA, **disabled** for Autoformer / FEDformer / TimesNet / FiLM (cuFFT needs fp32 at `seq_len`=96) |
| Model selection | lowest validation loss; that checkpoint is restored before test |
| Seeds | 0–4, seeding `random`, `numpy`, `torch`, `torch.cuda` |

No hyperparameter search was run for any model — see
[analysis/tuning_budget.md](analysis/tuning_budget.md).

## 6. Hardware and software

- **Training**: Kaggle notebook, single **NVIDIA Tesla T4** (16 GB), 12-hour
  session cap. The runner appends to a CSV after every run and skips any
  `(model, dataset, ablation, seed, …)` already present, so a session that dies
  at hour 12 loses one run rather than the batch.
- **Post-hoc analysis in `analysis/`**: Apple Silicon, macOS 25.6, CPU and MPS.
  Note that MPS raises an internal Metal assertion on the models that use
  `Tensor.unfold` for patching (PatchTST and XAI-MeteoFormer), so those were run
  on CPU.
- **Versions** (the local analysis environment, `.venv`):

  | Package | Version |
  |---|---|
  | Python | 3.9.6 |
  | torch | 2.8.0 |
  | numpy | 2.0.2 |
  | pandas | 2.3.3 |
  | scikit-learn | 1.6.1 |
  | scipy | 1.13.1 |

  `shap` is **not** a dependency: GradientSHAP is implemented directly in
  [src/xai.py](src/xai.py) with torch autograd.

  The Kaggle training environment's exact versions are **not recorded anywhere
  in the repository** — no `requirements.txt`, no `pip freeze`, no environment
  capture in [notebooks/kaggle_train.ipynb](notebooks/kaggle_train.ipynb). This
  is a genuine reproducibility gap and should be fixed before resubmission by
  adding a `pip freeze` cell to the notebook and committing the output.

- **Baselines**: [Time-Series-Library](https://github.com/thuml/Time-Series-Library),
  vendored in `Time-Series-Library/`. The adapter patches
  `layers/SelfAttention_Family.py` at import time to make the
  `reformer_pytorch` import optional; no baseline used here needs it.
  **The exact TSLib commit is not pinned** — another gap worth closing.

## 7. Compute budget

From the repaired `train_time_s`
([analysis/results_hygiene.md](analysis/results_hygiene.md)):

- **138 training runs, 18.8 GPU-hours** on the T4 in total.
- Per-model means are in [analysis/tuning_budget.md](analysis/tuning_budget.md).
- Post-hoc inference in `analysis/` (validation predictions for all 138
  checkpoints) adds no GPU time — it ran on the laptop.

## 8. Repository and artefacts

Repository: this working tree. `.gitignore` excludes `data/`, `checkpoints/`,
`predictions/`, `xai/`, `outputs/`, `Time-Series-Library/` and `paper/`, so the
tracked tree is source only. To reproduce end to end:

```bash
python src/data/prepare.py --dataset jena    --out data/processed
python src/data/prepare.py --dataset beijing --out data/processed --station Aotizhongxin
python src/train.py --dataset jena --models XAI-MeteoFormer DLinear PatchTST ... --seeds 0 1 2 3 4
python src/xai.py   --dataset jena --model XAI-MeteoFormer --ablation no_revin --seed 0
python src/report.py --out paper
```

Artefacts produced and retained locally (not in git):

| Artefact | Count | Size |
|---|---|---|
| `checkpoints/*.pt` | 138 | 1.8 GB |
| `predictions/*_pred.npy` (test) | 138 + 2 truth | 557 MB |
| `predictions_val/*_valpred.npy` | 138 + truth | — |
| `xai/*.npz` | 32 | 81 MB |

**Note on the artefact naming**: a checkpoint is
`{model}_{dataset}_{ablation}_s{seed}.pt`. After the `--norm_variant` flag was
added, a baseline trained in the non-historical normalization configuration gets
an extra `_norm{on,off}` suffix, so every checkpoint that already exists keeps
its name and stays valid.
