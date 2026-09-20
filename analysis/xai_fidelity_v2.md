# Explanation fidelity, all 11 models × 5 seeds (P0-3, P0-4)

Reviewer 1, Major #4 and #7. Produced by [fidelity_deterministic.py](analysis/fidelity_deterministic.py) (primary), [xai_backfill.py](analysis/xai_backfill.py) (appendix) and [xai_fidelity_v2.py](analysis/xai_fidelity_v2.py).

## Summary of what changed against the published table

- The published table set a 5-seed mean for our model against single-seed points for six baselines and had no entry for DLinear, Transformer, Informer and Autoformer. **Every model now has 5 seeds under identical conditions, on both datasets.**
- **SHAP now exists for every model.** It was missing for PatchTST, iTransformer and TFT because their normalisation divides in place (`x_enc /= stdev`); `src/xai.py` silently fell back to permutation importance. The harness rewrites that line out of place at runtime, forward pass bit-identical ([xai_patches.py](analysis/xai_patches.py)).
- **Baselines are explained in the normalization variant the main table reports**, selected on validation loss ([selection.py](analysis/selection.py)): Autoformer (both datasets) and Transformer (Beijing) with RevIN, LSTM (Jena) without it.
- **The primary metric is now deterministic.** See next section.

## The primary metric: deterministic fidelity

The permutation metric of `src/xai.py` has two sources of Monte-Carlo noise: the perturbation (a random time permutation of each perturbed channel) and the reference (a **single** random channel ordering). On one fixed checkpoint re-explained with five RNG seeds its sd is ~0.18 ([fidelity_estimator_noise.csv](analysis/fidelity_estimator_noise.csv)) — as large as the spread across five trained seeds, so at the current budget it cannot separate models closer than ~0.2.

The deterministic metric keeps the definition and removes both sources:

| | permutation metric (appendix) | deterministic metric (primary) |
|---|---|---|
| perturbation of a channel | random permutation along time | replacement by its own per-window mean |
| reference curve | one random channel ordering | k = 1: exact mean over channels; k ≥ 2: mean over 20 fixed orderings, shared by every model |
| model-agnostic ranking | permutation importance (random) | single-channel window-mean occlusion (deterministic) |
| SHAP | seeded by the model seed | same function and budget, torch seeded to 0 for every run |
| scoring | ks = 0,1,2,3,5,8; gain = mean(ranked − reference) over k>0; rel = gain / base MAE | identical |

**Determinism, verified.** The same checkpoint explained twice produces bit-identical importances and curves (`np.array_equal` on every array). Any spread across seeds in the tables below is therefore model variance, not estimator noise. The 20 reference orderings are a fixed sample, not the exact expectation over all orderings; being identical for every model, they cannot favour one.

Column meaning: **SHAP-ranked** is the fidelity of the model's GradientSHAP ranking — the cross-model comparison of explanation quality. **Occlusion-ranked** ranks channels by the same perturbation that scores them, so it is close to an upper reference, not a fair comparison. **ρ(SHAP, occl)** is the Spearman agreement of the two rankings.


---

## Jena

### Deterministic fidelity, mean ± sd over seeds

| Model | variant | n | SHAP-ranked | occlusion-ranked | ρ(SHAP, occl) |
|---|---|---|---|---|---|
| Crossformer | historical | 5 | 0.259 ± 0.028 | 0.265 ± 0.027 | 0.858 ± 0.101 |
| iTransformer | historical | 5 | 0.241 ± 0.006 | 0.256 ± 0.006 | 0.784 ± 0.088 |
| PatchTST | historical | 5 | 0.233 ± 0.004 | 0.243 ± 0.005 | 0.991 ± 0.003 |
| TimesNet | historical | 5 | 0.209 ± 0.019 | 0.256 ± 0.009 | 0.573 ± 0.168 |
| **MeteoFormer** | headline | 5 | 0.207 ± 0.007 | 0.211 ± 0.007 | 0.766 ± 0.130 |
| TFT | historical | 5 | 0.204 ± 0.038 | 0.232 ± 0.016 | 0.787 ± 0.098 |
| DLinear | historical | 5 | 0.186 ± 0.000 | 0.227 ± 0.001 | 0.747 ± 0.000 |
| LSTM | normoff | 5 | 0.146 ± 0.011 | 0.156 ± 0.009 | 0.841 ± 0.070 |
| Informer | historical | 5 | 0.135 ± 0.017 | 0.146 ± 0.018 | 0.774 ± 0.108 |
| Autoformer | normon | 5 | 0.125 ± 0.014 | 0.150 ± 0.013 | 0.598 ± 0.077 |
| Transformer | historical | 5 | 0.121 ± 0.017 | 0.135 ± 0.020 | 0.757 ± 0.106 |

**MeteoFormer, built-in attention ranking:** fidelity 0.012 ± 0.065; Spearman(attention, SHAP) 0.484 ± 0.335; Spearman(attention, occlusion) 0.371 ± 0.444.

### Paired tests against our model (SHAP-ranked, deterministic)

Paired by seed, paired t-test, Holm over the 10 baselines. Wilcoxon is not shown: with n = 5 its smallest two-sided p is 0.0625. Positive Δ = our model's SHAP ranking is more faithful.

| Baseline | Δ (ours − baseline) | paired t p | Holm | n |
|---|---|---|---|---|
| Crossformer | -0.052 | 0.0243 | 0.0729 | 5 |
| iTransformer | -0.034 | 0.000504 | 0.00453 **\*** | 5 |
| PatchTST | -0.026 | 0.00357 | 0.0143 **\*** | 5 |
| TimesNet | -0.002 | 0.779 | 1 | 5 |
| TFT | +0.003 | 0.868 | 1 | 5 |
| DLinear | +0.021 | 0.00282 | 0.0141 **\*** | 5 |
| LSTM | +0.061 | 0.00105 | 0.0072 **\*** | 5 |
| Informer | +0.072 | 0.00103 | 0.0072 **\*** | 5 |
| Autoformer | +0.082 | 0.000567 | 0.00454 **\*** | 5 |
| Transformer | +0.086 | 0.000422 | 0.00422 **\*** | 5 |

Our model's SHAP ranking is more faithful than 6 of 10 baselines on average; **7 of 10** differences survive Holm.


---

## Beijing (Aotizhongxin)

### Deterministic fidelity, mean ± sd over seeds

| Model | variant | n | SHAP-ranked | occlusion-ranked | ρ(SHAP, occl) |
|---|---|---|---|---|---|
| Crossformer | historical | 5 | 0.169 ± 0.016 | 0.185 ± 0.018 | 0.807 ± 0.082 |
| PatchTST | historical | 5 | 0.158 ± 0.011 | 0.176 ± 0.007 | 0.983 ± 0.010 |
| DLinear | historical | 5 | 0.144 ± 0.001 | 0.190 ± 0.002 | 0.760 ± 0.000 |
| iTransformer | historical | 5 | 0.142 ± 0.003 | 0.189 ± 0.003 | 0.524 ± 0.057 |
| **MeteoFormer** | headline | 5 | 0.130 ± 0.017 | 0.149 ± 0.012 | 0.669 ± 0.191 |
| TimesNet | historical | 5 | 0.109 ± 0.036 | 0.172 ± 0.007 | 0.655 ± 0.059 |
| TFT | historical | 5 | 0.104 ± 0.014 | 0.123 ± 0.016 | 0.744 ± 0.053 |
| LSTM | historical | 5 | 0.075 ± 0.013 | 0.167 ± 0.007 | 0.627 ± 0.103 |
| Autoformer | normon | 5 | 0.068 ± 0.007 | 0.077 ± 0.010 | 0.568 ± 0.139 |
| Informer | historical | 5 | 0.056 ± 0.014 | 0.072 ± 0.011 | 0.505 ± 0.121 |
| Transformer | normon | 5 | 0.042 ± 0.016 | 0.141 ± 0.019 | 0.321 ± 0.118 |

**MeteoFormer, built-in attention ranking:** fidelity 0.046 ± 0.092; Spearman(attention, SHAP) 0.266 ± 0.391; Spearman(attention, occlusion) 0.138 ± 0.403.

### Paired tests against our model (SHAP-ranked, deterministic)

Paired by seed, paired t-test, Holm over the 10 baselines. Wilcoxon is not shown: with n = 5 its smallest two-sided p is 0.0625. Positive Δ = our model's SHAP ranking is more faithful.

| Baseline | Δ (ours − baseline) | paired t p | Holm | n |
|---|---|---|---|---|
| Crossformer | -0.040 | 0.00311 | 0.0187 **\*** | 5 |
| PatchTST | -0.029 | 0.035 | 0.175 | 5 |
| DLinear | -0.014 | 0.135 | 0.404 | 5 |
| iTransformer | -0.012 | 0.159 | 0.404 | 5 |
| TimesNet | +0.021 | 0.168 | 0.404 | 5 |
| TFT | +0.026 | 0.0766 | 0.306 | 5 |
| LSTM | +0.055 | 0.000528 | 0.00422 **\*** | 5 |
| Autoformer | +0.061 | 0.00154 | 0.0107 **\*** | 5 |
| Informer | +0.074 | 0.000301 | 0.00301 **\*** | 5 |
| Transformer | +0.088 | 0.000341 | 0.00307 **\*** | 5 |

Our model's SHAP ranking is more faithful than 6 of 10 baselines on average; **5 of 10** differences survive Holm.


---

## P0-4: does the entropy regulariser buy faithfulness?

Isolated by two variants that differ in `lambda_ent` alone, on the published configuration: `no_revin` (λ=0.01, the headline) vs `no_revin+no_entropy` (λ=0), both RevIN off, 5 seeds, both datasets, paired by seed. Deterministic metric; the time-axis rows come from [occlusion_time.md](analysis/occlusion_time.md), which was deterministic from the start.

> Correction. An earlier version compared `no_entropy` (RevIN on) with `no_revin` (RevIN off) and concluded that the regulariser doubles the attention–occlusion agreement. That comparison changed RevIN too and is withdrawn.

### Jena

| Metric (n = 5) | λ=0.01 (headline) | λ=0 | Δ | paired t p | λ>0 better |
|---|---|---|---|---|---|
| fidelity, attention-ranked | 0.012 ± 0.065 | 0.103 ± 0.055 | -0.091 | 0.099 | 1/5 |
| Spearman(attention, SHAP) | 0.484 ± 0.335 | 0.301 ± 0.233 | +0.183 | 0.203 | 4/5 |
| Spearman(attention, occlusion) | 0.371 ± 0.444 | 0.468 ± 0.253 | -0.097 | 0.436 | 2/5 |
| fidelity, SHAP-ranked | 0.207 ± 0.007 | 0.201 ± 0.012 | +0.006 | 0.235 | 4/5 |
| fidelity, occlusion-ranked | 0.211 ± 0.007 | 0.207 ± 0.013 | +0.005 | 0.377 | 3/5 |
| time occlusion ρ(attention) | 0.578 ± 0.215 | 0.471 ± 0.508 | +0.107 | 0.712 | 2/5 |

Recency control on the same windows: ρ = 0.582 ± 0.205.

### Beijing (Aotizhongxin)

| Metric (n = 5) | λ=0.01 (headline) | λ=0 | Δ | paired t p | λ>0 better |
|---|---|---|---|---|---|
| fidelity, attention-ranked | 0.046 ± 0.092 | 0.105 ± 0.016 | -0.060 | 0.161 | 1/5 |
| Spearman(attention, SHAP) | 0.266 ± 0.391 | 0.816 ± 0.031 | -0.550 | 0.029 | 0/5 |
| Spearman(attention, occlusion) | 0.138 ± 0.403 | 0.771 ± 0.041 | -0.634 | 0.027 | 0/5 |
| fidelity, SHAP-ranked | 0.130 ± 0.017 | 0.130 ± 0.013 | -0.000 | 0.963 | 3/5 |
| fidelity, occlusion-ranked | 0.149 ± 0.012 | 0.142 ± 0.010 | +0.007 | 0.146 | 4/5 |
| time occlusion ρ(attention) | 0.622 ± 0.394 | 0.567 ± 0.248 | +0.055 | 0.719 | 2/5 |

Recency control on the same windows: ρ = 0.407 ± 0.401.

**Multiple comparisons.** 12 paired tests. Smallest raw p = 0.027 (Beijing (Aotizhongxin), Spearman(attention, occlusion)), Holm-adjusted 0.33. No difference survives the correction.

**Reading.** On the attention rows λ=0.01 comes out ahead in **12 of 40** seed-level comparisons. A regulariser that made attention more faithful would win most of them. The SHAP- and occlusion-ranked rows describe the model rather than its attention and should barely move.

**Conclusion for Major #7.** The entropy regulariser does not improve faithfulness; no difference survives correction and the direction on the attention rows is against it. The claim that it "lifts the fidelity/stability numbers" (also in the docstring of [src/xm_models/xai_meteoformer.py](src/xm_models/xai_meteoformer.py)) is not supported. What it demonstrably does is sparsify the attention ([results_hygiene.md](analysis/results_hygiene.md)); if kept, describe it as a sparsity prior only.


---

## What does move faithfulness: RevIN

`full` (RevIN on) vs the headline `no_revin` (RevIN off), 5 seeds each, paired by seed, deterministic metric.

### Jena

| Metric | RevIN on (full) | RevIN off (headline) | Δ | paired t p | on > off |
|---|---|---|---|---|---|
| fidelity, attention-ranked | 0.181 ± 0.079 | 0.012 ± 0.065 | +0.169 | 0.001 | 5/5 |
| Spearman(attention, SHAP) | 0.839 ± 0.060 | 0.484 ± 0.335 | +0.354 | 0.072 | 5/5 |
| Spearman(attention, occlusion) | 0.514 ± 0.129 | 0.371 ± 0.444 | +0.144 | 0.429 | 2/5 |
| fidelity, SHAP-ranked | 0.240 ± 0.013 | 0.207 ± 0.007 | +0.033 | 0.006 | 5/5 |
| fidelity, occlusion-ranked | 0.260 ± 0.014 | 0.211 ± 0.007 | +0.048 | 0.004 | 5/5 |

### Beijing (Aotizhongxin)

| Metric | RevIN on (full) | RevIN off (headline) | Δ | paired t p | on > off |
|---|---|---|---|---|---|
| fidelity, attention-ranked | 0.098 ± 0.035 | 0.046 ± 0.092 | +0.053 | 0.398 | 4/5 |
| Spearman(attention, SHAP) | 0.425 ± 0.216 | 0.266 ± 0.391 | +0.159 | 0.521 | 3/5 |
| Spearman(attention, occlusion) | 0.299 ± 0.141 | 0.138 ± 0.403 | +0.162 | 0.448 | 3/5 |
| fidelity, SHAP-ranked | 0.136 ± 0.022 | 0.130 ± 0.017 | +0.006 | 0.687 | 3/5 |
| fidelity, occlusion-ranked | 0.177 ± 0.013 | 0.149 ± 0.012 | +0.028 | 0.038 | 4/5 |


---

## Appendix: the permutation metric of `src/xai.py`

Same models, variants, windows and SHAP budget; perturbation by random time permutation and a single random reference ordering. **Caveat: on one fixed checkpoint this metric has sd ≈ 0.18 from its own Monte-Carlo noise at the current budget** — as large as the between-seed spread — so differences below ~0.2 are not interpretable. Kept for continuity with the published table, whose sd of 0.037 came from reusing one RNG stream across checkpoints.

| One fixed checkpoint, RNG seed 0–4 | SHAP-ranked | permutation-ranked |
|---|---|---|
| Crossformer | 0.378 ± 0.176 | 0.369 ± 0.163 |
| **MeteoFormer** | 0.413 ± 0.180 | 0.384 ± 0.174 |

### Jena, permutation metric

| Model | n | SHAP-ranked | permutation-ranked | ρ(SHAP, perm) |
|---|---|---|---|---|
| Crossformer | 5 | 0.420 ± 0.194 | 0.413 ± 0.182 | 0.834 ± 0.080 |
| **MeteoFormer** | 5 | 0.401 ± 0.189 | 0.382 ± 0.178 | 0.779 ± 0.114 |
| iTransformer | 5 | 0.281 ± 0.127 | 0.299 ± 0.128 | 0.823 ± 0.063 |
| DLinear | 5 | 0.266 ± 0.106 | 0.301 ± 0.108 | 0.749 ± 0.003 |
| PatchTST | 5 | 0.265 ± 0.115 | 0.276 ± 0.117 | 0.991 ± 0.003 |
| TFT | 5 | 0.186 ± 0.110 | 0.221 ± 0.097 | 0.759 ± 0.054 |
| LSTM | 5 | 0.166 ± 0.075 | 0.182 ± 0.086 | 0.836 ± 0.093 |
| Informer | 5 | 0.161 ± 0.078 | 0.172 ± 0.084 | 0.820 ± 0.027 |
| TimesNet | 5 | 0.146 ± 0.093 | 0.175 ± 0.096 | 0.389 ± 0.256 |
| Transformer | 5 | 0.135 ± 0.068 | 0.150 ± 0.071 | 0.753 ± 0.122 |
| Autoformer | 5 | 0.090 ± 0.080 | 0.161 ± 0.065 | 0.410 ± 0.108 |

### Beijing (Aotizhongxin), permutation metric

| Model | n | SHAP-ranked | permutation-ranked | ρ(SHAP, perm) |
|---|---|---|---|---|
| iTransformer | 5 | 0.273 ± 0.084 | 0.318 ± 0.084 | 0.533 ± 0.040 |
| Crossformer | 5 | 0.269 ± 0.079 | 0.283 ± 0.079 | 0.854 ± 0.022 |
| DLinear | 5 | 0.257 ± 0.070 | 0.294 ± 0.071 | 0.760 ± 0.000 |
| **MeteoFormer** | 5 | 0.235 ± 0.095 | 0.251 ± 0.094 | 0.664 ± 0.136 |
| PatchTST | 5 | 0.224 ± 0.067 | 0.241 ± 0.067 | 0.986 ± 0.013 |
| TimesNet | 5 | 0.129 ± 0.074 | 0.147 ± 0.068 | 0.683 ± 0.084 |
| LSTM | 5 | 0.109 ± 0.052 | 0.137 ± 0.049 | 0.667 ± 0.099 |
| TFT | 5 | 0.101 ± 0.039 | 0.108 ± 0.044 | 0.685 ± 0.086 |
| Informer | 5 | 0.079 ± 0.045 | 0.098 ± 0.048 | 0.608 ± 0.180 |
| Transformer | 5 | 0.051 ± 0.039 | 0.108 ± 0.024 | 0.135 ± 0.172 |
| Autoformer | 5 | 0.035 ± 0.026 | 0.083 ± 0.042 | 0.160 ± 0.173 |


---

## Verdict for P0-3 and P0-4

**Jena.** Our model ranks **5 of 11** by SHAP-ranked fidelity (0.207 ± 0.007). Against Crossformer: Δ = -0.052, Holm p = 0.0729 — not significant. Significantly more faithful than ours: PatchTST, iTransformer. Built-in attention: 0.012 ± 0.065 — indistinguishable from zero, and now with a tight interval rather than the wide one the permutation metric produced. Removing the estimator noise shrank the seed sd of our own score from 0.189 to 0.007.

**Beijing (Aotizhongxin).** Our model ranks **5 of 11** by SHAP-ranked fidelity (0.130 ± 0.017). Against Crossformer: Δ = -0.040, Holm p = 0.0187 — **Crossformer is significantly more faithful**. Significantly more faithful than ours: Crossformer. Built-in attention: 0.046 ± 0.092 — indistinguishable from zero, and now with a tight interval rather than the wide one the permutation metric produced. Removing the estimator noise shrank the seed sd of our own score from 0.095 to 0.017.

**What this changes.** Under the noisy permutation metric our model looked second, just behind Crossformer, with seed sds of ~0.19 that made every ranking meaningless. With the noise removed the ordering is resolved and it is less flattering: our model sits mid-table on both datasets. The honest claim is no longer "most faithful" but "comparable to the strongest baselines by SHAP-ranked fidelity, ahead of the weaker half, and behind Crossformer".

**The built-in attention remains the weak point** and is now firmly so: its fidelity is ~0 with a small interval, and it is far below the same model's SHAP ranking. The defensible framing is that the model is explainable by post-hoc attribution, not that its attention is an explanation.

**Two findings are unchanged by the metric swap**, which is the best evidence that they are real: the entropy regulariser does not buy faithfulness (P0-4 above), and RevIN does — the configuration selected for accuracy, `no_revin`, has markedly less faithful attention than `full` (Jena: attention-ranked fidelity 0.181 vs 0.012, paired p = 0.001, 5 of 5 seeds).
