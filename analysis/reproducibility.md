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

Two environments produced the results, and they are **not** interchangeable.
Every number in the paper comes from one of them; the table below says which.

### 6.1 Training — Kaggle (GPU)

Everything that trains a model ran on Kaggle notebooks: a single **NVIDIA
Tesla T4** (16 GB) for the published runs, and **2 × Tesla T4** for the
resubmission runs, which drive one process per card. 12-hour session cap; the
runner appends to a CSV after every run and skips any
`(model, dataset, ablation, seed, norm_variant, width, loss, aug_block)`
already present, so a session that dies at hour 12 loses one run, not the batch.

| | Value |
|---|---|
| GPU | NVIDIA Tesla T4, 14 911 MiB |
| Python | 3.12.13 |
| torch | 2.10.0+cu128 |
| CUDA runtime | 12.8 |
| cuDNN | 91002 |
| OS | Linux 6.12.90, glibc 2.35 |
| TSLib | `4e938a1767106324dd753b2a44832bf870a0252e` (pinned, asserted) |

Captured automatically by every run into `analysis/environment.json` and
`analysis/requirements_frozen.txt` (`pip freeze`, 939 packages) — this is
Reviewer 1 Minor #8. **The runs behind the originally published tables predate
that capture and their exact environment is not recorded**; the code, the
pinned TSLib commit and the hardware model are the same, and the paper should
say so rather than imply a full capture.

### 6.2 Post-hoc analysis — laptop (CPU / MPS)

Everything in `analysis/` that only *reads* checkpoints and predictions ran
locally. No training, no GPU hours.

| | Value |
|---|---|
| Hardware | Apple Silicon, macOS 26.6 (Darwin 25.6), CPU and MPS |
| Python | 3.9.6 |
| torch | 2.8.0 |
| numpy / pandas / scipy | 2.0.2 / 2.3.3 / 1.13.1 |
| scikit-learn | 1.6.1 |

MPS raises an internal Metal assertion on `Tensor.unfold`, used for patching by
PatchTST and XAI-MeteoFormer. The analysis harnesses replace it at runtime with
an equivalent stack of slices
([mps_unfold_fix.py](analysis/mps_unfold_fix.py); outputs agree with CPU to
~2e-6, gradients to ~6e-7 relative), so everything runs on MPS. `src/` is not
modified.

That workaround was written part-way through the analysis, so a few results
were computed on CPU before it existed and the rest on MPS afterwards — marked
in 6.3. The mixture is harmless precisely because the two paths were checked
against each other at the tolerances above, but it is recorded rather than
smoothed over. `shap` is **not** a dependency: GradientSHAP is implemented directly
in [src/xai.py](src/xai.py) with torch autograd.

### 6.3 Which result came from which environment

| Result | File | Environment |
|---|---|---|
| Main accuracy tables, ablations, normalization variants, Crossformer widths, window sweep, MSE and outage-augmentation variants | `analysis/results_clean.csv`, `analysis/results_variants.csv`, `analysis/results_seqlen*.csv` | **Kaggle, T4** |
| Test predictions, written by the training run itself | `predictions/*_pred.npy` | **Kaggle, T4** |
| Validation predictions, re-run from the checkpoints afterwards | `predictions_val/*_valpred.npy` ([dump_val_preds.py](analysis/dump_val_preds.py)) | laptop, CPU / MPS |
| Significance (Diebold–Mariano, paired tests) | `analysis/significance*.md/.csv` | laptop, CPU (reads predictions) |
| Explanation fidelity, deterministic and permutation | `analysis/xai_fidelity_v2.md`, `analysis/fidelity_det.csv` | laptop, MPS |
| Time-axis occlusion | `analysis/occlusion_time.md` | laptop: CPU for `no_revin` and `no_entropy`, MPS for the rest |
| Attention stability among correlated predictors | `analysis/attention_stability.md` | laptop, CPU — **no model is run**, it reads `xai/*.npz` |
| Target-specificity of the temporal pooling | `analysis/temporal_pooling_target_specificity.md` | **no computation** — answered from the source code |
| Missing-input robustness, block gaps | `analysis/missing_robustness.md`, `analysis/block_missing.md` | laptop, MPS, except the PatchTST and XAI-MeteoFormer rows, computed on CPU before the `unfold` workaround (noted in `block_missing.md`) |
| Zero-shot cross-station transfer | `analysis/cross_station.md` | laptop, MPS |
| Error decomposition, dispersion calibration | `analysis/error_decomposition.md`, `analysis/variance_calibration.md` | laptop, CPU (numpy only) |
| Frost events, per-target and per-horizon tables | `analysis/frost_events.md`, `analysis/per_target_horizon.md` | laptop, CPU |

Two models are not bit-reproducible by construction, on either machine:
`Informer` samples keys with `torch.randint` inside `ProbAttention`'s forward
pass, and `TimesNet` runs an FFT. Re-running their inference moves the MAE in
the fourth decimal (measured: 3.3e-3 for Informer, 2.1e-4 for TimesNet, both
< 0.3 % relative). Every other model reproduced its published in-domain MAE to
1e-7 through a different harness
([analysis/cross_station.md](analysis/cross_station.md), control section).

- **Baselines**: [Time-Series-Library](https://github.com/thuml/Time-Series-Library),
  vendored in `Time-Series-Library/`. The adapter patches
  `layers/SelfAttention_Family.py` at import time to make the
  `reformer_pytorch` import optional; no baseline used here needs it.
  **TSLib is pinned** to `4e938a1767106324dd753b2a44832bf870a0252e` in every
  notebook (a shallow fetch of that one commit, plus an assert that fails the
  run if it drifts).

## 7. Compute budget

From `train_time_s` of every training run in the repository
([analysis/results_hygiene.md](analysis/results_hygiene.md) explains the repair
applied to the older rows):

| Set | Runs | GPU-hours |
|---|---|---|
| published configuration (`analysis/results_clean.csv`) | 300 | 36.4 |
| training-recipe variants — MSE, outage augmentation (`analysis/results_variants.csv`) | 230 | 31.6 |
| window sweep, L = 24/48/192 (`analysis/results_seqlen*.csv`) | 30 | 4.2 |
| **total** | **560** | **72.2** |

Per-model means are in [analysis/tuning_budget.md](analysis/tuning_budget.md).
Everything in `analysis/` — validation and test inference, SHAP, occlusion,
robustness harnesses, cross-station transfer — adds **no GPU time**: it ran on
the laptop of §6.2.

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
