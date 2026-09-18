# Table regeneration and the reach of the corrupted columns

## 1. Regeneration diff: no paper number changes

`paper/tables/*.tex` regenerated from
[analysis/results_clean.csv](analysis/results_clean.csv) and diffed against the
previous files. **All nine tables came back byte-identical.**

```
identical: ablation.tex            identical: main_beijing_aotizhongxin.tex
identical: events_beijing_*.tex    identical: main_jena.tex
identical: events_jena.tex         identical: significance_beijing_*.tex
identical: fidelity_beijing_*.tex  identical: significance_jena.tex
identical: fidelity_jena.tex
```

The reason is that [src/report.py](src/report.py) was **already** repairing the
column shift at load time, in `load_results()`, and already de-duplicating on
`(model, dataset, ablation, seed)` with `keep="last"`. So the corruption never
reached the tables.

Two hardening changes were still made:

- The shift detector was `d["params"] < 1000` — a heuristic that happens to work
  because `lambda_ent` lands in the `params` cell. It is now
  `d["seq_len"] != 96`, which is exact: `seq_len` is 96 in every run ever made,
  and in a rotated row that cell holds the epoch count.
- `--results` now defaults to `analysis/results_clean.csv`.

## 2. Where the corrupted values could have reached

**Inside generated artefacts: nowhere.** `report.py` emits only `MAE`, `RMSE`,
`R2` and `best_val` into tables; `params`, `train_time_s` and `infer_time_s` are
never written to any table or to `paper/summary.txt`. `best_val` does reach the
ablation table, and it was repaired on load.

**Outside generated artefacts: anything typed by hand from the raw CSVs.** The
manuscript `.tex`/`.docx` is **not in this repository** — only `paper/tables/`
and `paper/figures/` are — so the prose cannot be audited from here. These are
the claims to check by hand, with the correct values:

| Claim type | Corrupted reading in the raw CSV | Correct value |
|---|---|---|
| "Crossformer / TFT / TimesNet has *N* parameters" | **0** | Crossformer 1.70 M, TFT 3.64 M, TimesNet 1.19 M (Jena) |
| "training took *N* minutes/hours" | 3–4 orders of magnitude too large (Crossformer read as 470 h/run) | Crossformer 22.2 min/run (Jena), 12.3 (Beijing) |
| "inference time" | held `best_val` instead | Crossformer 8.36 s (Jena) |
| "best validation loss" | held `infer_time_s` (≈6–12 instead of ≈0.15) | see table below |
| total compute | — | **18.8 GPU-h over 138 runs** |

Affected rows were 36 of 81 in `jena_results.csv` (Crossformer, TFT, TimesNet,
and all Jena ablations of XAI-MeteoFormer) and 15 of 60 in
`beijing_results.csv`.

### Correct reference values

| Model | params (Jena) | train min/run (Jena) | train min/run (Beijing) | infer s (Jena) |
|---|---|---|---|---|
| XAI-MeteoFormer | 1 892 513 | 10.6 | 5.1 | 6.6 |
| Autoformer | 2 677 267 | 23.4 | 5.9 | 12.3 |
| Crossformer | 1 695 908 | 22.2 | 12.3 | 8.4 |
| DLinear | 4 656 | 1.0 | 0.7 | 1.9 |
| Informer | 2 867 475 | 5.8 | 2.5 | 6.4 |
| LSTM | 834 694 | 1.1 | 0.6 | 2.4 |
| PatchTST | 1 657 880 | 20.3 | 6.0 | 9.8 |
| TFT | 3 640 676 | 16.1 | 7.1 | 11.9 |
| TimesNet | 1 186 507 | 15.7 | 6.9 | 9.1 |
| Transformer | 2 670 099 | 4.1 | 1.8 | 6.0 |
| iTransformer | 1 611 032 | 1.6 | 0.7 | 1.9 |

`params` for LSTM is 834 694 **with** RevIN (38 = 2×19 affine parameters); the
standard LSTM without it is 834 656. See
[analysis/tuning_symmetry_audit.md](analysis/tuning_symmetry_audit.md).

## 3. Ablation table now states the seed count

`paper/tables/ablation.tex` gained an `$n$ seeds` column. The variants are not
equally sampled and the table previously did not say so:

| Variant | n seeds | Val. loss | MAE | RMSE | R² |
|---|---|---|---|---|---|
| no revin | **5** | 0.1437 | 3.025 ± 0.031 | 5.267 ± 0.046 | 0.682 ± 0.005 |
| no multiscale | **3** | 0.1521 | 3.122 ± 0.028 | 5.445 ± 0.052 | 0.664 ± 0.003 |
| no cls | **3** | 0.1537 | 3.132 ± 0.032 | 5.495 ± 0.032 | 0.659 ± 0.003 |
| no fusion | **3** | 0.1557 | 3.211 ± 0.021 | 5.586 ± 0.042 | 0.651 ± 0.004 |
| no entropy | **3** | 0.1561 | 3.186 ± 0.036 | 5.529 ± 0.055 | 0.655 ± 0.004 |
| full | **5** | 0.1564 | 3.195 ± 0.019 | 5.536 ± 0.039 | 0.656 ± 0.004 |
| no temp attn | **3** | 0.1565 | 3.194 ± 0.020 | 5.541 ± 0.027 | 0.655 ± 0.002 |
| no var attn | **3** | 0.1572 | 3.198 ± 0.017 | 5.561 ± 0.014 | 0.654 ± 0.003 |

Item 3 of the work list (top up every ablation to 5 seeds on both datasets)
removes this column's reason to exist; until those runs are done, the column is
the honest way to present the table.

## 4. `results_repaired.csv` renamed

`results_repaired.csv` → `SUPERSEDED_results_repaired_jena_partial.csv`. It was
a Jena-only partial fix (76 rows, `no_revin` at 3 seeds) that is easy to mistake
for the authoritative table. It is kept because it independently corroborates
the repair: all 76 of its rows agree with
[analysis/results_clean.csv](analysis/results_clean.csv) on every column.

The authoritative table is **`analysis/results_clean.csv`** — 138 rows, one per
existing checkpoint, no duplicate keys, `seq_len == 96` throughout.

## 5. New tables written

Beyond the regenerated ones:

| File | Content |
|---|---|
| `significance_{ds}.tex` | **replaced** — Diebold–Mariano instead of unpaired Welch |
| `significance_channels_{ds}.tex` | new — DM differential per target channel |
| `per_target_{mae,rmse,r2}_{ds}.tex` | new — per-target accuracy, all models |
| `per_horizon_{mae,rmse,r2}_{ds}.tex` | new — h = 1, 6, 12, 24, all models |
| `events_{ds}.tex` | rewritten by [analysis/frost_events.py](analysis/frost_events.py) |
