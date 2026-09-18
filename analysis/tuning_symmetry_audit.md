# Audit: was the normalization choice symmetric across models?

Source-level audit of [src/baselines/tslib_adapter.py](src/baselines/tslib_adapter.py)
and the `Time-Series-Library` checkout actually used, cross-checked against what
the runs recorded. No training, no inference.

## Direct answer

**Yes. The proposed model received a validation-selected RevIN on/off choice
that no baseline received.** `XAI-MeteoFormer` was trained in both
configurations, 5 seeds, both datasets, and the better one (`no_revin`) was
promoted to the headline row of `paper/tables/main_jena.tex` on validation loss.
Every baseline ran in exactly one normalization configuration, fixed by its
source code, never varied and never selected.

There is a second, subtler asymmetry: the normalization configuration the
baselines were locked into was **not uniform**. Five of them ran with
per-window instance normalization always on, five with none at all.

## Per-model table

`instance norm` = per-input-window mean/std normalization of the encoder input
with de-normalization of the output (RevIN in all but name).

| Model | Normalization in the path actually executed | Configurable? | Value used | Chosen on validation? |
|---|---|---|---|---|
| **XAI-MeteoFormer** | **RevIN, affine, learnable** ([src/xm_layers/revin.py](src/xm_layers/revin.py)) | **yes — `use_revin`** | **both on and off trained** | **yes** |
| PatchTST | instance norm, non-affine, hardcoded in `forecast()` | no | always on | no |
| iTransformer | instance norm, non-affine, hardcoded in `forecast()` | no | always on | no |
| TimesNet | instance norm, non-affine, hardcoded in `forecast()` | no | always on | no |
| TFT | instance norm, non-affine, hardcoded in `forecast()` | no | always on | no |
| LSTM | **RevIN, affine, learnable — added by our adapter** | no | always on | no |
| DLinear | none | no | always off | no |
| Crossformer | none | no | always off | no |
| Transformer | none | no | always off | no |
| Informer | none — `long_forecast()` has no normalization | no | always off | no |
| Autoformer | none — series decomposition only, not instance norm | no | always off | no |

Evidence for each claim:

- **PatchTST / iTransformer / TimesNet / TFT**: the block
  `means = x_enc.mean(1, keepdim=True).detach() ... x_enc /= stdev`, commented
  "Normalization from Non-stationary Transformer", sits unguarded at the top of
  `forecast()` with the inverse applied to `dec_out`. There is no `if` around
  it and no config attribute controls it.
- **DLinear / Crossformer / Transformer**: `forecast()` goes straight from
  input to embedding. Grep for `means|stdev|revin|normaliz` in their model files
  returns nothing.
- **Informer**: it has *two* forecast methods. `short_forecast()` normalizes;
  `long_forecast()` does not. The adapter sets `task_name="long_term_forecast"`,
  so the normalizing path was never executed.
- **Autoformer**: `torch.mean(x_enc, dim=1)` is the trend initialiser for the
  decomposition decoder, not an input normalization — the encoder input is
  untouched and nothing is de-normalized on output.
- **LSTM**: `SimpleRNN` in the adapter instantiates `RevIN(n_channels)` and
  applies it. This is our layer, not a property of an LSTM baseline; the
  adapter's docstring states the intent ("withholding it here would stack the
  deck against the classic baseline"), but it is still an addition beyond the
  standard baseline and should be disclosed as one.

### The `use_norm` flag is dead code here

`make_configs()` sets `use_norm=1` ([src/baselines/tslib_adapter.py:141](src/baselines/tslib_adapter.py#L141)),
which looks like a normalization switch applied to all baselines. It is not.
Grepping the TSLib checkout, `use_norm` is read by exactly two models —
`TimeXer` and `TimeMixer` — and **neither was run**: TimeXer is not in
`TSLIB_MODELS` at all, and TimeMixer is registered but appears in no result row
and has no checkpoint. For all 10 baselines that were actually run, `use_norm`
has no effect whatsoever. Do not cite it in the paper as evidence that
normalization was held constant.

## How much this matters, honestly

It matters, but not in the direction that is most damaging:

- The four baselines that **do** get instance normalization (PatchTST,
  iTransformer, TimesNet, TFT) all rank *below* our model on both datasets. The
  asymmetry did not manufacture our wins over them.
- **Crossformer — the model we lose RMSE to on Jena — runs with no instance
  normalization at all.** So the gap we are trying to explain was measured
  against an *un-normalized* competitor. This cuts both ways: we cannot be
  accused of beating Crossformer by giving ourselves a normalization it was
  denied, but we also cannot claim the comparison is tight, because giving
  Crossformer RevIN might well make it *better* and widen the RMSE gap.
- The defensible claim after this audit is narrow: "RevIN on/off was selected on
  validation for the proposed model; baselines were run in their published
  normalization configuration." That is a legitimate sentence, but it must be
  written down, because a reviewer reading `use_norm=1` would otherwise assume
  symmetry that does not exist.

Note also that six of the eight Jena ablations have only 3 seeds, while `full`
and `no_revin` have 5. The ablation table mixes 3-seed and 5-seed rows without
saying so.

## Cost of making it symmetric

From `train_time_s` in [analysis/results_clean.csv](analysis/results_clean.csv)
(the raw CSVs' timing column is corrupted by a column shift — see
[analysis/results_hygiene.md](analysis/results_hygiene.md); these are the
repaired values). Mean wall-clock per run on the Kaggle T4:

| Model | Beijing (min) | Jena (min) | 10 runs (h) |
|---|---|---|---|
| Crossformer | 12.3 | 22.2 | 2.87 |
| Autoformer | 5.9 | 23.4 | 2.44 |
| PatchTST | 6.0 | 20.3 | 2.19 |
| TFT | 7.1 | 16.1 | 1.94 |
| TimesNet | 6.9 | 15.7 | 1.88 |
| Informer | 2.5 | 5.8 | 0.69 |
| Transformer | 1.8 | 4.1 | 0.49 |
| iTransformer | 0.7 | 1.6 | 0.19 |
| LSTM | 0.6 | 1.1 | 0.15 |
| DLinear | 0.7 | 1.0 | 0.14 |
| *XAI-MeteoFormer* | *5.1* | *10.6* | *1.31* |

One extra normalization variant = 5 seeds × 2 datasets = **10 runs per model**.

| Tier | Scope | Runs | GPU-hours |
|---|---|---|---|
| **A** | Add RevIN to the 5 un-normalized baselines (DLinear, **Crossformer**, Transformer, Informer, Autoformer) | 50 | **6.6** |
| **B** | Remove instance norm from the 4 normalized baselines (PatchTST, iTransformer, TimesNet, TFT) | 40 | **6.2** |
| **C** | Drop RevIN from LSTM (restore the standard baseline) | 10 | **0.1** |
| | **all three** | **100** | **13.0** |

For scale, the entire existing result set is 138 runs and **18.8 GPU-h**. Full
symmetry costs less than one extra pass over the study — about two Kaggle T4
sessions.

### Recommended scope

**Tier A alone, 6.6 GPU-h**, and within it Crossformer is the one that actually
decides anything (2.9 h on its own). Rationale:

- Tier A is the honest test of the reviewer's likely objection: "you gave
  yourself RevIN and denied it to the models you beat." Giving the
  un-normalized baselines the *same* RevIN layer we use answers it directly.
- Tier B is weaker. Instance normalization is part of the published PatchTST /
  iTransformer / TimesNet / TFT architectures; stripping it produces a model the
  authors never proposed, and a reviewer can reasonably object that this is
  handicapping a baseline rather than tuning it.
- Tier C is nearly free and removes a genuine, self-inflicted inconsistency.

Be aware of the downside before spending the quota: Tier A may show that
Crossformer with RevIN is better still, which would widen the Jena RMSE gap
rather than close it. That is a real possibility and the audit cannot predict
the outcome — but finding it now is better than having a reviewer find it, and
the per-channel result in [analysis/error_decomposition.md](analysis/error_decomposition.md)
(we win temperature decisively, lose humidity) does not depend on it.

Implementation note: Tier A needs a small adapter change, not a training change
— wrap the baseline core in the existing `RevIN` the way `SimpleRNN` already
does, gated by a new flag — implemented as `--norm_variant {on,off}`, see
[norm_symmetry_runs.md](analysis/norm_symmetry_runs.md) — so both variants of
each baseline are separate runs under the resume key.
