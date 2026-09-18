# Results hygiene: the 8 `no_revin` rows, the column shift, and the entropy term

No training. Reproduced by
[analysis/repair_results_csv.py](analysis/repair_results_csv.py) and
[analysis/entropy_gradient_check.py](analysis/entropy_gradient_check.py).
Originals are untouched; the cleaned table is
[analysis/results_clean.csv](analysis/results_clean.csv).

## 1. What the 8 rows actually are

**They are 5 runs, not 8, and they all used the same `lambda_ent = 0.01`.**
There was never a λ sweep. The apparent second value is a misread column.

Rows 40–75 of `jena_results.csv` were written under an older row schema in which
`lambda_ent` sat *before* `params` instead of at the end. `append_row()` in
[src/train.py:127](src/train.py#L127) concatenates on column *names*, so rows
written by the current code align correctly — but these rows predate that fix and
carry a one-position rotation across the whole block `params … lambda_ent`.

The rotation is detectable with no ambiguity: **`seq_len` is 96 in every run ever
made**, and in a rotated row the `seq_len` cell holds the epoch count (10–23).

Raw, for `no_revin`:

| idx | params | train_time_s | infer_time_s | best_val | epochs_run | seq_len | lr | lambda_ent |
|---|---|---|---|---|---|---|---|---|
| 55 | 0.01 | 1892475 | 614.5 | 6.47 | 0.141985 | 19 | 2.0 | 0.0005 |
| 76 | 1892475 | 651.6 | 7.12 | 0.141985 | 19 | 96 | 0.0005 | 0.0100 |

Read row 55 one column to the left and it *is* row 76: `lambda_ent=0.01`,
`params=1892475`, `best_val=0.141985`, `epochs_run=19`, `seq_len=96`,
`lr=0.0005`. The value that looked like `lambda_ent=0.0005` is the **learning
rate**; the true λ, 0.01, was sitting in the `params` column.

So:

- **Cause of the identical metrics: they are literally the same runs**, logged
  twice in two different column layouts. `MAE`, `RMSE`, `MSE` and `R²` differ by
  exactly `0.0` between the pairs — not "suspiciously close", bit-identical.
- Seeds 3 and 4 appear only in the newer layout, which is why the count is 8 and
  not 10.
- No log or checkpoint inspection was needed; the CSV is self-diagnosing via
  `seq_len`. This is independently confirmed by `results_repaired.csv`, an
  earlier partial fix sitting in the repo: all 76 of its rows agree with my
  repaired values on **every** column.

### Scope of the damage

| File | rows | column-shifted | duplicate runs |
|---|---|---|---|
| `jena_results.csv` | 81 | 36 (idx 40–75) | 3 |
| `beijing_results.csv` | 60 | 15 | 0 |

Affected models: Crossformer, TFT, TimesNet, XAI-MeteoFormer (all Jena ablations).

**The metric columns (`MAE` onward) are NOT affected.** Verified directly
against `predictions/*_pred.npy`: recomputed MAE/RMSE match the CSV to 5
decimals in both layouts. The rotation is confined to the metadata block. So
every accuracy number in the paper is correct.

**What *was* wrong and is worth knowing:**

- `train_time_s` is garbage in shifted rows — it holds `params`. Any efficiency
  claim computed from the raw CSV is wrong by 3–4 orders of magnitude (Crossformer
  reads as 470 h/run instead of 22 min). The repaired totals: the whole study is
  **18.8 GPU-h over 138 runs**.
- `params` is garbage too — it holds λ, so Crossformer/TFT/TimesNet all read as
  **0 parameters**. If a parameter-count column appears in the paper, check it.
- `best_val` in shifted rows holds `infer_time_s`. The ablation table's "Val.
  loss" column happens to be correct anyway, because `report.py` read the
  epoch-count cell, which is where the real `best_val` landed — but that is luck,
  not design.

### The cleaned file

`analysis/results_clean.csv`: 138 rows, exactly one per existing checkpoint, zero
duplicate `(model, dataset, ablation, seed)` keys, `seq_len == 96` throughout.

```
jena_results.csv:    81 rows, 36 column-shifted, 3 duplicate runs dropped -> 78
beijing_results.csv: 60 rows, 15 column-shifted, 0 duplicate runs dropped -> 60
-> analysis/results_clean.csv (138 rows)
```

Note it also makes the seed asymmetry explicit: Jena `full` and `no_revin` have
5 seeds, the other six ablations have 3.

## 2. Does the entropy term actually do anything?

Yes, on all three checks.

**Gradient reaches the parameters.** `beta` is
`softmax(einsum(h_t, variable_query) / sqrt(d))`
([xai_meteoformer.py:253-256](src/xm_models/xai_meteoformer.py#L253-L256)) and
`var_entropy` is computed from it at
[line 279](src/xm_models/xai_meteoformer.py#L279). Backpropagating `var_entropy`
alone on a real batch:

| Variant | entropy at init | `requires_grad` | params with nonzero grad | total ‖grad‖ |
|---|---|---|---|---|
| `full` (`use_var_attn=True`) | 2.9443 | True | **48 of 54** | 0.612 |
| `no_var_attn` | 2.9444 | **False** | 0 | — |

The gradient flows through the encoder, not just the query: the largest
contributions are `layers.{0,1}.ff.3.weight` and `attn.out_proj.weight`.

For `no_var_attn`, `beta` is a constant `1/N` tensor, so `var_entropy` is a
detached constant with no gradient path. That is correct behaviour — but it
means `no_var_attn` silently also disables the entropy regulariser. The ablation
removes two things at once, and the paper should say so.

**The term is not negligible.** On a real batch, Huber ≈ 0.331 and entropy ≈
2.944, so at `lambda_ent = 0.01` the entropy contributes **8.2 %** of the total
loss. (At the phantom 0.0005 it would have been 0.44 %.)

**The trained weights show the effect.** Comparing `full` (λ=0.01) against
`no_entropy` (λ=0) checkpoints, per seed, and measuring realised attention
entropy on a validation batch:

| seed | weights identical | max ‖Δ variable_query‖ | entropy, `full` | entropy, `no_entropy` |
|---|---|---|---|---|
| 0 | no | 0.162 | 2.777 | 2.898 |
| 1 | no | 0.195 | 2.606 | 2.827 |
| 2 | no | 0.191 | 2.437 | 2.895 |
| **mean** | | | **2.606** | **2.873** |

With ln(N) = ln(19) = 2.944 as the uniform-attention ceiling, `no_entropy` ends
up essentially uniform (2.873) while `full` is measurably sparser (2.606). The
regulariser does what it claims. The effect is real but modest — worth stating
with the numbers rather than as a qualitative claim.

## 3. What to do

1. Use `analysis/results_clean.csv` as the source for any table that touches
   `params`, `train_time_s`, `infer_time_s` or `best_val`. Regenerate
   `paper/tables/*` from it and diff against the current `.tex`.
2. Delete `results_repaired.csv` or rename it — it is a Jena-only partial fix
   that is easy to mistake for the authoritative file.
3. In the ablation table, mark which rows have 3 seeds and which have 5.
4. State in the text that `no_var_attn` also removes the entropy regulariser.
