# Normalization symmetry: the `--norm_variant` flag, and what still needs a GPU

Item 2 of the work list. The code is written, verified and backward-compatible.
**The training runs themselves are not done — they need the Kaggle T4.**

## 1. What changed

Three files. Full diffs in [analysis/diffs/](analysis/diffs/):
[tslib_adapter.diff](analysis/diffs/tslib_adapter.diff),
[revin.diff](analysis/diffs/revin.diff),
[train.diff](analysis/diffs/train.diff).

**Only the normalization changed.** Nothing in the diff touches model
construction, config values, the loss, the optimiser, the schedule, the split or
the scaler. The functional edits in the adapter are exactly:

| Change | Effect |
|---|---|
| `use_norm=1` annotated `# UNUSED by every baseline we run` | documents the dead flag; still passed so the config stays valid if TimeXer/TimeMixer are added |
| new `UNNORMALIZED` / `SELF_NORMALIZING` sets | names which models have no per-window normalization and which hardcode it |
| `SimpleRNN(..., use_revin=bool)` | lets LSTM run **without** our RevIN, i.e. the textbook baseline |
| `BaselineWrapper(..., revin=None)` | optional external RevIN, applied at input and inverted at output |
| `build_baseline(..., norm_variant="on"/"off")` | the switch |

The RevIN added to a baseline is the **same layer the proposed model uses**
([src/xm_layers/revin.py](src/xm_layers/revin.py)) — not a reimplementation — so
the comparison is like for like.

`SELF_NORMALIZING` (PatchTST, iTransformer, TimesNet, TFT) is deliberately
untouched: `norm_variant=on` prints "already self-normalizing, no-op" and
changes nothing. Per the agreed scope, those four stay in their published
configurations.

## 2. Backward compatibility, verified

The flag reproduces the historical configuration exactly. Parameter counts are
the proof — RevIN adds exactly `2 × n_channels` affine parameters (38 on Jena):

| Model | `off` | `on` | recorded in `results_clean.csv` |
|---|---|---|---|
| LSTM | 834 656 | **834 694** | **834 694** → LSTM historically ran with RevIN |
| Crossformer | **1 695 908** | 1 695 946 | **1 695 908** → historically without |
| DLinear | **4 656** | 4 694 | **4 656** → historically without |
| PatchTST | 1 657 880 | 1 657 880 (no-op) | 1 657 880 |

So `historical = "on" if model == "LSTM" else "off"`, which is what the tagging
logic in [src/train.py](src/train.py) now encodes: only the *new* variant gets a
`_norm{on,off}` suffix, so **every existing checkpoint keeps its name and stays
loadable**.

Verified end to end by re-running test inference on existing checkpoints with
the patched code:

| Run | recomputed MAE/RMSE | stored MAE/RMSE |
|---|---|---|
| Crossformer_jena_full_s0 | 3.0843 / 5.1592 | 3.0843 / 5.1592 |
| LSTM_jena_full_s0 | 3.3804 / 5.5745 | 3.3804 / 5.5745 |
| DLinear_jena_full_s0 | 3.4168 / 5.7509 | 3.4168 / 5.7509 |
| XAI-MeteoFormer_jena_no_revin_s0 | 2.9851 / 5.2302 | 2.9851 / 5.2302 |
| XAI-MeteoFormer_jena_full_s0 | 3.1819 / 5.5414 | 3.1819 / 5.5414 |

All exact. The refactor is behaviour-preserving.

## 3. The runs that still need a GPU

| Model | Existing variant | Variant to add | Runs | GPU-h |
|---|---|---|---|---|
| Crossformer | off | **on** | 10 | 2.87 |
| Autoformer | off | **on** | 10 | 2.44 |
| Informer | off | **on** | 10 | 0.69 |
| Transformer | off | **on** | 10 | 0.49 |
| DLinear | off | **on** | 10 | 0.14 |
| LSTM | on | **off** | 10 | 0.15 |
| | | **total** | **60** | **≈6.8** |

(5 seeds × 2 datasets each; times from the repaired `train_time_s`.)

Command:

```bash
# the five un-normalised baselines get RevIN
python src/train.py --dataset jena --norm_variant on \
    --models DLinear Crossformer Transformer Informer Autoformer --seeds 0 1 2 3 4
python src/train.py --dataset beijing_aotizhongxin --norm_variant on \
    --models DLinear Crossformer Transformer Informer Autoformer --seeds 0 1 2 3 4
# LSTM loses ours, returning to the standard baseline
python src/train.py --dataset jena                --norm_variant off --models LSTM --seeds 0 1 2 3 4
python src/train.py --dataset beijing_aotizhongxin --norm_variant off --models LSTM --seeds 0 1 2 3 4
```

`norm_variant` is part of the resume key, so a 12-hour session that dies mid-way
resumes without redoing finished runs, and none of these collide with the
existing rows.

**Selection rule, to apply after the runs finish**: per (model, dataset), pick
the variant with the lower mean validation loss across the 5 seeds — the same
rule that selected `no_revin` for our model — and report that variant in the
main table, with the other variant in an appendix. Selection on validation only;
test is scored once.

## 4. What to expect, stated in advance

This experiment can go against us, and it is worth writing the prediction down
before running it so the outcome is not rationalised afterwards.

[analysis/error_decomposition.md](analysis/error_decomposition.md) showed RevIN
*hurts* our model on Jena, uniformly and most at high amplitude. If that is a
property of the data rather than of our architecture, adding RevIN to
Crossformer should hurt Crossformer too — and our position improves. If instead
it is an interaction with our patching/multiscale front end, Crossformer with
RevIN may improve and widen the Jena RMSE gap.

Either outcome is publishable. The first supports a claim about meteorological
data; the second is an honest limitation that is far better found by us than by
a reviewer. What is **not** defensible is the current state, where our model got
a validation-selected normalization choice and no baseline did.
