# Block gaps: an operational outage, and a second external check (P0-5)

Produced by [analysis/block_missing.py](analysis/block_missing.py) and [analysis/block_missing_report.py](analysis/block_missing_report.py). Inference only on the existing checkpoints; 11 models × 5 seeds × both datasets, on the same test subsample as the per-cell experiment ({'beijing_aotizhongxin': 3448, 'jena': 2782} windows).

## Method

A contiguous block of **16 hours is removed from every channel** of the 96-hour input window — a station outage, not scattered sensor dropouts. Two placements with an **identical missing fraction** (16/96 = 16.7% of input cells):

- **A — end of the window** (t = 80…96, the most recent 16 h)
- **B — start of the window** (t = 0…16, the oldest 16 h)

Repair uses the same rule as the per-cell experiment: linear interpolation along time over the surviving indices, per channel. For a block at the end of the window there is nothing to interpolate towards, so this degenerates to **holding the last valid value** (t = 79) for 16 hours — exactly what an operational forward-fill does after an outage. For a block at the start it holds the first valid value backwards.

This is also a second external test of [analysis/occlusion_time.md](analysis/occlusion_time.md), which found that our model uses essentially only the last 16 hours. If so, A must hurt far more than B.


---

## Jena

### MAE, mean ± sd over 5 seeds

| Model | intact | A: last 16 h lost | B: first 16 h lost | Δ A | Δ B |
|---|---|---|---|---|---|
| Autoformer | 3.903±0.047 | 5.012±0.112 | 4.053±0.058 | **+28.4%** | +3.8% |
| TimesNet | 3.313±0.015 | 5.005±0.117 | 3.400±0.033 | **+51.1%** | +2.6% |
| iTransformer | 3.300±0.013 | 5.146±0.058 | 3.372±0.020 | **+55.9%** | +2.2% |
| PatchTST | 3.352±0.019 | 5.230±0.087 | 3.434±0.039 | **+56.0%** | +2.4% |
| Crossformer | 3.110±0.020 | 4.920±0.035 | 3.130±0.023 | **+58.2%** | +0.6% |
| Informer | 3.457±0.065 | 5.523±0.123 | 3.602±0.052 | **+59.8%** | +4.2% |
| DLinear | 3.427±0.008 | 5.610±0.020 | 3.493±0.010 | **+63.7%** | +1.9% |
| **MeteoFormer** | 3.028±0.031 | 5.251±0.289 | 3.045±0.030 | **+73.4%** | +0.6% |
| Transformer | 3.452±0.067 | 6.032±0.244 | 3.437±0.056 | **+74.7%** | -0.4% |
| TFT | 3.388±0.066 | 6.046±0.307 | 3.444±0.080 | **+78.5%** | +1.7% |
| LSTM | 3.317±0.045 | 5.976±0.115 | 3.318±0.045 | **+80.1%** | +0.0% |

Across all 55 (model, seed) pairs, losing the **last** 16 h costs **+61.8%** MAE on average, losing the **first** 16 h costs **+1.8%**. Paired Wilcoxon A vs B: p = 1.1e-10. A is worse than B for **55 of 55** pairs.

### Does first place survive an outage in the last 16 h?

| Rank | intact | A: last 16 h lost |
|---|---|---|
| 1 | **MeteoFormer** 3.028 | Crossformer 4.920 |
| 2 | Crossformer 3.110 | TimesNet 5.005 |
| 3 | iTransformer 3.300 | Autoformer 5.012 |
| 4 | TimesNet 3.313 | iTransformer 5.146 |
| 5 | LSTM 3.317 | PatchTST 5.230 |
| 6 | PatchTST 3.352 | **MeteoFormer** 5.251 |
| 7 | TFT 3.388 | Informer 5.523 |
| 8 | DLinear 3.427 | DLinear 5.610 |
| 9 | Transformer 3.452 | LSTM 5.976 |
| 10 | Informer 3.457 | Transformer 6.032 |
| 11 | Autoformer 3.903 | TFT 6.046 |

Under an outage in the last 16 h our model falls from **1st to 6th** by MAE.

### Per target, Δ% of MAE under A (last 16 h lost)

| Model | T | RH | P | WS |
|---|---|---|---|---|
| Autoformer | +27 | +23 | +46 | +11 |
| Crossformer | +47 | +52 | +104 | +23 |
| DLinear | +71 | +70 | +51 | +36 |
| Informer | +37 | +65 | +81 | +28 |
| LSTM | +58 | +92 | +86 | +37 |
| PatchTST | +50 | +50 | +94 | +24 |
| TFT | +69 | +87 | +78 | +36 |
| TimesNet | +46 | +45 | +84 | +25 |
| Transformer | +58 | +89 | +66 | +34 |
| **MeteoFormer** | +73 | +67 | +111 | +28 |
| iTransformer | +48 | +50 | +91 | +29 |


---

## Beijing (Aotizhongxin)

### MAE, mean ± sd over 5 seeds

| Model | intact | A: last 16 h lost | B: first 16 h lost | Δ A | Δ B |
|---|---|---|---|---|---|
| Autoformer | 5.347±0.112 | 6.007±0.094 | 5.493±0.082 | **+12.4%** | +2.7% |
| Transformer | 4.895±0.071 | 6.196±0.129 | 4.938±0.052 | **+26.6%** | +0.9% |
| TimesNet | 4.426±0.033 | 6.110±0.100 | 4.567±0.072 | **+38.1%** | +3.2% |
| Informer | 4.935±0.092 | 7.033±0.050 | 5.172±0.089 | **+42.5%** | +4.8% |
| PatchTST | 4.369±0.046 | 6.394±0.045 | 4.469±0.043 | **+46.3%** | +2.3% |
| iTransformer | 4.250±0.017 | 6.249±0.076 | 4.354±0.013 | **+47.0%** | +2.4% |
| Crossformer | 4.248±0.095 | 6.338±0.178 | 4.295±0.085 | **+49.2%** | +1.1% |
| DLinear | 4.361±0.008 | 6.587±0.034 | 4.487±0.009 | **+51.0%** | +2.9% |
| LSTM | 4.343±0.034 | 6.893±0.116 | 4.432±0.039 | **+58.7%** | +2.0% |
| **MeteoFormer** | 4.150±0.039 | 6.637±0.244 | 4.167±0.040 | **+59.9%** | +0.4% |
| TFT | 4.676±0.177 | 8.070±0.112 | 4.814±0.172 | **+72.7%** | +3.0% |

Across all 55 (model, seed) pairs, losing the **last** 16 h costs **+45.9%** MAE on average, losing the **first** 16 h costs **+2.3%**. Paired Wilcoxon A vs B: p = 1.1e-10. A is worse than B for **55 of 55** pairs.

### Does first place survive an outage in the last 16 h?

| Rank | intact | A: last 16 h lost |
|---|---|---|
| 1 | **MeteoFormer** 4.150 | Autoformer 6.007 |
| 2 | Crossformer 4.248 | TimesNet 6.110 |
| 3 | iTransformer 4.250 | Transformer 6.196 |
| 4 | LSTM 4.343 | iTransformer 6.249 |
| 5 | DLinear 4.361 | Crossformer 6.338 |
| 6 | PatchTST 4.369 | PatchTST 6.394 |
| 7 | TimesNet 4.426 | DLinear 6.587 |
| 8 | TFT 4.676 | **MeteoFormer** 6.637 |
| 9 | Transformer 4.895 | LSTM 6.893 |
| 10 | Informer 4.935 | Informer 7.033 |
| 11 | Autoformer 5.347 | TFT 8.070 |

Under an outage in the last 16 h our model falls from **1st to 8th** by MAE.

### Per target, Δ% of MAE under A (last 16 h lost)

| Model | T | RH | P | WS |
|---|---|---|---|---|
| Autoformer | -1 | +16 | +10 | +7 |
| Crossformer | +30 | +51 | +77 | +17 |
| DLinear | +68 | +52 | +42 | +28 |
| Informer | +21 | +50 | +43 | +21 |
| LSTM | +43 | +63 | +65 | +23 |
| PatchTST | +50 | +44 | +61 | +21 |
| TFT | +74 | +80 | +49 | +31 |
| TimesNet | +19 | +39 | +60 | +15 |
| Transformer | +12 | +27 | +44 | +8 |
| **MeteoFormer** | +63 | +57 | +83 | +19 |
| iTransformer | +43 | +48 | +51 | +21 |


---

## Verdict

**1. The occlusion finding is confirmed — and it is not specific to our model.** Every one of the 11 models is hurt far more by losing the most recent 16 hours than by losing the oldest 16 hours of the same window, at the same missing fraction. Relying on the recent past is a property of hourly weather forecasting at a 24 h horizon (persistence is a strong predictor), not an idiosyncrasy of our architecture. That makes occlusion_time.md *less* damaging than it first looked: "the model uses the last 16 hours" is what every model here does.

**2. Our model is the most recency-concentrated of all.** It is among the most damaged by losing the recent block and the least damaged by losing the old one. Jena: +73% vs +0.6%; Beijing: +60% vs +0.4%. It extracts almost nothing from hours 0–80 of its window, more so than any baseline.

**3. First place does not survive an outage in the last 16 hours.** On both datasets our model drops out of the top by MAE once the most recent block is lost and forward-filled. This is the operationally relevant failure mode — a station outage — and it must be reported, not only the per-cell result where every model looked robust.

**4. What it means for the paper.** The per-cell experiment ([missing_robustness.md](analysis/missing_robustness.md)) and this one answer different questions and should be reported together: scattered dropouts are harmless for everyone because interpolation repairs them; a recent outage is severe for everyone and worst for us. Combined with the window-length sweep (P0-2, GPU track), this points to a concrete, testable design consequence — our model's 96-hour context is largely unused, and its accuracy advantage is bought with maximal dependence on the freshest observations.

## Caveat on devices

For PatchTST and our model, part of the Jena rows were computed on CPU before the MPS `unfold` workaround and the rest on MPS after it. The two paths agree to ~2·10⁻⁶ on outputs and ~6·10⁻⁷ relative on gradients (verified on real checkpoints, see [analysis/xai_patches.py](analysis/xai_patches.py)); the mix has no material effect on any number here.
