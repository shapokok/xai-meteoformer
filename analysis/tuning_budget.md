# Tuning budget (Reviewer 2)

Reconstructed from the code and every recorded run:
[results_clean.csv](analysis/results_clean.csv) (300 runs) plus the window
sweep in `analysis/results_seqlen{24,48,192}.csv` (30 runs). All times are the
repaired `train_time_s` on a Kaggle T4.

> **Revision.** The first version of this file described the *published* study:
> 138 runs, no search at all, Crossformer possibly under-provisioned. The
> resubmission work added validation-based selection for several models and ran
> Crossformer at full width. This version supersedes the old one; both old
> claims are resolved below.

## Direct answer

The published results used **no hyperparameter search for any model**: one
configuration each, library defaults or capacity-matched settings. The
resubmission adds a **small, validation-only selection** for the proposed model
and for six baselines. Every selection uses the same rule — **the lowest mean
validation loss over 5 seeds** — and the test set is scored once, after
selection. Nothing is chosen on test.

The budgets are **not equal**, and the paper must say so plainly: the proposed
model was selected among 17 configurations, each baseline among 1–3.

## Per model

Parameter counts are for Jena (19 input channels); Beijing differs by a few
dozen parameters in the input and RevIN layers.

| Model | Configs explored | Runs | GPU-h | Params, selected config | Params across explored configs | What was varied | Selected on validation |
|---|---|---|---|---|---|---|---|
| **XAI-MeteoFormer** | **17** | 170 | 21.7 | **1 892 475** | 1 755 512 – 1 892 513 | RevIN on/off; 6 ablations of `full`; 6 ablations of `no_revin`; window 24/48/96/192 | `no_revin`, L=96 (headline, fixed by decision); validation would pick L=24 — see below |
| Crossformer | 3 | 20 | 5.0 | 1 695 908 | 1 695 908 – **10 608 420** | width 128×128 / 256×512 / 256×1024 | **128×128** (val 0.1489 < 0.1505 < 0.1524) |
| Autoformer | 2 | 20 | 4.2 | 2 677 305 | 2 677 267 – 2 677 305 | `norm_variant` off/on | **on** (both datasets) |
| DLinear | 2 | 20 | 0.4 | 4 656 | 4 656 – 4 694 | `norm_variant` off/on | off |
| Informer | 2 | 20 | 1.6 | 2 867 475 | 2 867 475 – 2 867 513 | `norm_variant` off/on | off |
| Transformer | 2 | 20 | 1.0 | 2 670 099 | 2 670 099 – 2 670 137 | `norm_variant` off/on | off on Jena, **on** on Beijing |
| LSTM | 2 | 20 | 0.5 | 834 656 | 834 656 – 834 694 | `norm_variant` off/on (on = our RevIN) | **off** on Jena, on (historical) on Beijing |
| PatchTST | 1 | 10 | 2.2 | 1 657 880 | — | — | published configuration |
| TFT | 1 | 10 | 1.9 | 3 640 676 | — | — | published configuration |
| TimesNet | 1 | 10 | 1.9 | 1 186 507 | — | — | published configuration |
| iTransformer | 1 | 10 | 0.2 | 1 611 032 | — | — | published configuration |
| **total** | | **330** | **40.6** | | | | |

The selected configurations are what the main table now reports
(`analysis/selection.py`, `paper/tables/main_*.tex`); the historical ones are
in the appendix (`main_historical_*.tex`). RevIN adds exactly `2 × channels`
affine parameters (38 on Jena), which is the whole difference between the two
normalization variants of every baseline. The proposed model's range comes from
its ablations: removing the multiscale block or the fusion gate drops it to
1.76 M.

5 seeds × 2 datasets per configuration throughout. PatchTST, TFT, TimesNet and
iTransformer normalise inside their own published forward pass, so they have no
normalization variant to select (see
[tuning_symmetry_audit.md](analysis/tuning_symmetry_audit.md)).

Normalization selection differs from the historical configuration in 4 of 10
(model, dataset) pairs — Autoformer on both datasets and Transformer on Beijing
move to `on`, LSTM on Jena moves to `off`. By decision, the main table now
reports the selected variants and the historical ones move to the appendix. The proposed model still beats every
selected variant by MAE with Diebold–Mariano p < 1e-7 after Holm
([norm_selection.md](analysis/norm_selection.md)).

## Shared configuration

Every model: `pred_len`=24, `n_layers`=2, `dropout`=0.2, `lr`=5e-4, batch 64,
AdamW (`weight_decay`=1e-4), OneCycleLR (`pct_start`=0.3), max 40 epochs,
early stopping on validation loss (`patience`=6), gradient clipping at 1.0,
Huber loss (`delta`=1.0) on globally standardised targets. `d_model`=256
unless overridden: Crossformer 128/128 (published; 256 was also run), TimesNet
32/32, TFT 128 — `MODEL_OVERRIDES` in
[src/baselines/tslib_adapter.py](src/baselines/tslib_adapter.py). `seq_len`=96
for everything except the window sweep of the proposed model.

## The two findings that change the paper

**1. Crossformer was not under-provisioned.** The earlier caveat — that it ran at
128 wide while we ran at 256 — is resolved. At full width it has 4–6× our
parameter count and is **worse** on validation *and* on test
([crossformer_fullwidth.md](analysis/crossformer_fullwidth.md)):

| Crossformer, Jena | params | val loss | MAE | RMSE |
|---|---|---|---|---|
| 128×128 (published) | 1.70 M | **0.1489** | **3.107** | **5.166** |
| 256×512 | 7.98 M | 0.1505 | 3.184 | 5.241 |
| 256×1024 | 10.61 M | 0.1524 | 3.175 | 5.237 |

The published comparison stands, and the capacity question has a one-line
answer for the reviewer.

**2. Validation prefers a 24-hour window for the proposed model — the headline
was not changed, per instruction.** ([window_sweep.md](analysis/window_sweep.md))

| L (h) | val loss, Jena | MAE, Jena | RMSE, Jena | val loss, Beijing | MAE, Beijing |
|---|---|---|---|---|---|
| **24** | **0.1389** | **2.940** | **5.147** | **0.1580** | **4.058** |
| 48 | 0.1417 | 2.994 | 5.205 | 0.1590 | 4.107 |
| 96 (headline) | 0.1437 | 3.025 | 5.267 | 0.1592 | 4.151 |
| 192 | 0.1457 | 3.041 | 5.272 | 0.1648 | 4.234 |

Under the same rule that picked `no_revin`, validation would pick L=24 on both
datasets. At L=24 the Jena RMSE (5.147) is **below Crossformer's 5.166** — the
one metric we currently lose. This is the mechanism found by
[occlusion_time.md](analysis/occlusion_time.md) and
[block_missing.md](analysis/block_missing.md): the model uses the last ~16 h
and the rest of a 96 h window is noise to it.

Adopting L=24 is a legitimate validation-based choice, but it is a **decision
for the authors**, not an analysis step, and it has a price in budget
asymmetry: the proposed model would then be selected over window length too,
which no baseline was. The honest options are (a) keep L=96 and report the sweep
as mechanism evidence, or (b) adopt L=24 and give the baselines the same window
sweep before claiming the RMSE result.

## How to present this

The defensible sentence is: *the published baselines were run in their
published configurations; in the resubmission, normalization (five baselines)
and width (Crossformer) were additionally selected on validation, and the
proposed model's architecture variants were selected on validation under the
same rule.* Then state the counts — 17 configurations for ours, 1–3 per
baseline — rather than leaving the reviewer to reconstruct them.
