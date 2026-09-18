# Explanation fidelity, all 11 models × 5 seeds (P0-3, P0-4)

Reviewer 1, Major #4 and #7. Produced by [analysis/xai_backfill.py](analysis/xai_backfill.py) and [analysis/xai_fidelity_v2.py](analysis/xai_fidelity_v2.py).

## What changed against the published table

The published fidelity table set a **5-seed mean** for our model against **single-seed points** for six baselines, and had **no entry at all** for DLinear, Transformer, Informer and Autoformer. Everything is now recomputed from scratch under identical conditions for all 11 models:

- 5 seeds per model, both datasets — 110 runs, none failed;
- 2 560 explained test windows (40 batches × 64), the same windows for every model;
- GradientSHAP with 16 samples on 10 batches for every model;
- the same fidelity k-grid, physical channels only;
- the permutation RNG seeded by the model seed, so seed *i* of every model sees the same permutation stream — the per-seed pairing in the tests below is a real pairing, not an index coincidence.

**SHAP now exists for every model.** It was missing for PatchTST, iTransformer and TFT in the published table because their normalisation divides in place (`x_enc /= stdev`), which breaks input gradients; `src/xai.py` caught the error and silently fell back to permutation importance. The harness rewrites that one line out of place at runtime (forward pass bit-identical, checked with `torch.equal`); see [analysis/xai_patches.py](analysis/xai_patches.py). Neither `src/` nor the pinned TSLib is modified on disk.

## First: the fidelity estimator is dominated by its own noise

To separate model differences from Monte-Carlo noise, **one** checkpoint per model (seed 0) was explained five times with RNG seeds 0–4 and nothing else changed ([fidelity_estimator_noise.csv](analysis/fidelity_estimator_noise.csv), reproducer [fidelity_estimator_noise.sh](analysis/fidelity_estimator_noise.sh)):

| Model (one fixed checkpoint) | SHAP-ranked | permutation-ranked | attention-ranked |
|---|---|---|---|
| Crossformer | 0.378 ± 0.176 | 0.369 ± 0.163 | — |
| **MeteoFormer** | 0.413 ± 0.180 | 0.384 ± 0.174 | 0.030 ± 0.190 |

The spread on a **single fixed model** (sd ≈ 0.17–0.18) is as large as the spread across five independently trained seeds in the tables below (0.18–0.19). Almost all of the seed-to-seed variance is estimator noise, not model difference. Two consequences:

1. **The published sd of 0.037 for our model was an artefact.** The published runs evidently used the same RNG stream for every checkpoint: for seed 0 the old and new permutation rankings agree exactly (ρ = 1.000), for seed 1 they do not (ρ = 0.816). The same random draw repeated five times hides the estimator's noise, and the published mean of 0.523 is one draw of it.
2. **At the current budget (2 560 windows, one permutation per batch) the metric cannot separate models whose fidelity differs by less than roughly 0.2.** Rankings inside that band are noise.

The fix is a lower-variance estimator, applied uniformly: replace the random permutation of the top-k channels with a deterministic replacement by each channel's own window mean — the same device used for the time axis in [occlusion_time.md](analysis/occlusion_time.md) — which has no Monte-Carlo noise at all. That is a change of metric definition, so it is proposed here, not applied.

`fidelity gain (rel)` = relative error increase when the top-ranked channels are perturbed, beyond what a random ranking produces. Higher = the ranking identifies channels the model actually depends on.


---

## Jena

### Fidelity by ranking method, mean ± sd over 5 seeds

| Model | n | SHAP-ranked | permutation-ranked | Spearman(SHAP, perm) |
|---|---|---|---|---|
| Crossformer | 5 | 0.420 ± 0.194 | 0.413 ± 0.182 | 0.834 ± 0.080 |
| **MeteoFormer** | 5 | 0.401 ± 0.189 | 0.382 ± 0.178 | 0.779 ± 0.114 |
| iTransformer | 5 | 0.281 ± 0.127 | 0.299 ± 0.128 | 0.823 ± 0.063 |
| DLinear | 5 | 0.266 ± 0.106 | 0.301 ± 0.108 | 0.749 ± 0.003 |
| PatchTST | 5 | 0.265 ± 0.115 | 0.276 ± 0.117 | 0.991 ± 0.003 |
| TFT | 5 | 0.186 ± 0.110 | 0.221 ± 0.097 | 0.759 ± 0.054 |
| LSTM | 5 | 0.177 ± 0.077 | 0.183 ± 0.079 | 0.738 ± 0.089 |
| Informer | 5 | 0.161 ± 0.078 | 0.172 ± 0.084 | 0.820 ± 0.027 |
| TimesNet | 5 | 0.146 ± 0.093 | 0.175 ± 0.096 | 0.389 ± 0.256 |
| Transformer | 5 | 0.135 ± 0.068 | 0.150 ± 0.071 | 0.753 ± 0.122 |
| Autoformer | 5 | 0.095 ± 0.133 | 0.178 ± 0.173 | 0.682 ± 0.069 |

**MeteoFormer, built-in attention ranking:** fidelity 0.040 ± 0.230; Spearman(attention, SHAP) 0.484 ± 0.335; Spearman(attention, permutation) 0.343 ± 0.405.

### Paired tests against our model (SHAP-ranked fidelity)

Paired by seed. Paired t-test, Holm-corrected over the 10 baselines; Wilcoxon signed-rank alongside. **With n = 5 the smallest two-sided Wilcoxon p attainable is 0.0625**, so it cannot reach 0.05 whatever the effect; it is reported for completeness. Positive Δ = our model's SHAP ranking is more faithful.

| Baseline | Δ (ours − baseline) | paired t p | Holm | Wilcoxon p |
|---|---|---|---|---|
| Crossformer | -0.019 | 0.274 | 0.274 | 0.312 |
| iTransformer | +0.120 | 0.0184 | 0.0674 | 0.0625 |
| DLinear | +0.134 | 0.0257 | 0.0674 | 0.0625 |
| PatchTST | +0.136 | 0.0156 | 0.0674 | 0.0625 |
| TFT | +0.214 | 0.0105 | 0.0674 | 0.0625 |
| LSTM | +0.223 | 0.0125 | 0.0674 | 0.0625 |
| Informer | +0.240 | 0.00963 | 0.0674 | 0.0625 |
| TimesNet | +0.255 | 0.00466 | 0.0419 **\*** | 0.0625 |
| Transformer | +0.266 | 0.00843 | 0.0674 | 0.0625 |
| Autoformer | +0.305 | 0.000636 | 0.00636 **\*** | 0.0625 |

Our model's SHAP ranking is more faithful than 9 of 10 baselines on average; **2 of 10** differences survive Holm.

### Published table vs recomputed (permutation-ranked)

| Model | published (seeds) | recomputed, 5 seeds |
|---|---|---|
| Autoformer | — (absent) | 0.178 ± 0.173 |
| Crossformer | 0.512 (2) | 0.413 ± 0.182 |
| DLinear | — (absent) | 0.301 ± 0.108 |
| Informer | — (absent) | 0.172 ± 0.084 |
| LSTM | 0.234 (2) | 0.183 ± 0.079 |
| PatchTST | 0.380 (2) | 0.276 ± 0.117 |
| TFT | 0.353 (2) | 0.221 ± 0.097 |
| TimesNet | 0.257 (2) | 0.175 ± 0.096 |
| Transformer | — (absent) | 0.150 ± 0.071 |
| **MeteoFormer** | 0.523 (5) | 0.382 ± 0.178 |
| iTransformer | 0.412 (2) | 0.299 ± 0.128 |


---

## Beijing (Aotizhongxin)

### Fidelity by ranking method, mean ± sd over 5 seeds

| Model | n | SHAP-ranked | permutation-ranked | Spearman(SHAP, perm) |
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
| Transformer | 5 | 0.049 ± 0.024 | 0.060 ± 0.027 | 0.560 ± 0.088 |
| Autoformer | 5 | -0.011 ± 0.059 | 0.075 ± 0.063 | 0.309 ± 0.415 |

**MeteoFormer, built-in attention ranking:** fidelity 0.112 ± 0.126; Spearman(attention, SHAP) 0.265 ± 0.390; Spearman(attention, permutation) 0.168 ± 0.452.

### Paired tests against our model (SHAP-ranked fidelity)

Paired by seed. Paired t-test, Holm-corrected over the 10 baselines; Wilcoxon signed-rank alongside. **With n = 5 the smallest two-sided Wilcoxon p attainable is 0.0625**, so it cannot reach 0.05 whatever the effect; it is reported for completeness. Positive Δ = our model's SHAP ranking is more faithful.

| Baseline | Δ (ours − baseline) | paired t p | Holm | Wilcoxon p |
|---|---|---|---|---|
| iTransformer | -0.038 | 0.0317 | 0.121 | 0.0625 |
| Crossformer | -0.034 | 0.0302 | 0.121 | 0.0625 |
| DLinear | -0.022 | 0.165 | 0.33 | 0.188 |
| PatchTST | +0.011 | 0.512 | 0.512 | 0.625 |
| TimesNet | +0.106 | 0.00291 | 0.0273 **\*** | 0.0625 |
| LSTM | +0.127 | 0.00383 | 0.0306 **\*** | 0.0625 |
| TFT | +0.134 | 0.00938 | 0.0563 | 0.0625 |
| Informer | +0.156 | 0.00273 | 0.0273 **\*** | 0.0625 |
| Transformer | +0.186 | 0.0047 | 0.0329 **\*** | 0.0625 |
| Autoformer | +0.246 | 0.0116 | 0.058 | 0.0625 |

Our model's SHAP ranking is more faithful than 7 of 10 baselines on average; **4 of 10** differences survive Holm.

### Published table vs recomputed (permutation-ranked)

| Model | published (seeds) | recomputed, 5 seeds |
|---|---|---|
| Autoformer | — (absent) | 0.075 ± 0.063 |
| Crossformer | 0.365 (2) | 0.283 ± 0.079 |
| DLinear | — (absent) | 0.294 ± 0.071 |
| Informer | — (absent) | 0.098 ± 0.048 |
| LSTM | 0.184 (3) | 0.137 ± 0.049 |
| PatchTST | 0.310 (2) | 0.241 ± 0.067 |
| TFT | 0.161 (2) | 0.108 ± 0.044 |
| TimesNet | 0.203 (2) | 0.147 ± 0.068 |
| Transformer | — (absent) | 0.060 ± 0.027 |
| **MeteoFormer** | 0.322 (5) | 0.251 ± 0.094 |
| iTransformer | 0.390 (2) | 0.318 ± 0.084 |


---

## P0-4: does the entropy regulariser buy faithfulness? (Jena)

**Control.** `no_entropy` is defined relative to `full` (`ABLATIONS["no_entropy"] = {}` in [src/train.py](src/train.py)), so it has **RevIN on**. The headline model is `no_revin`. Comparing `no_entropy` with the headline would change two things at once — the regulariser *and* RevIN. The only clean comparison is **`full` vs `no_entropy`**, which differ in `lambda_ent` alone (0.01 vs 0). The headline is shown for context only.

> Correction. An earlier version of this analysis compared `no_entropy` with `no_revin` and concluded that the regulariser roughly doubles the attention–occlusion agreement. That comparison was confounded by RevIN and is withdrawn; the numbers below replace it.

`no_entropy` has **3 seeds**; the clean comparison uses the matched seeds [0, 1, 2]. Seeds 3–4 and all of Beijing are queued in the GPU track, so this is **provisional**.

| Metric | full (λ=0.01) | no_entropy (λ=0) | Δ full − no_entropy | headline no_revin (context) |
|---|---|---|---|---|
| fidelity, attention-ranked | 0.197 ± 0.183 | 0.284 ± 0.078 | -0.088 | 0.040 ± 0.230 |
| Spearman(attention, SHAP) | 0.821 ± 0.071 | 0.837 ± 0.146 | -0.015 | 0.484 ± 0.335 |
| Spearman(attention, perm) | 0.462 ± 0.138 | 0.596 ± 0.254 | -0.135 | 0.343 ± 0.405 |
| fidelity, SHAP-ranked | 0.355 ± 0.186 | 0.365 ± 0.178 | -0.009 | 0.401 ± 0.189 |
| fidelity, permutation-ranked | 0.368 ± 0.181 | 0.374 ± 0.178 | -0.006 | 0.382 ± 0.178 |

Occlusion along time, same code for all three:

| Variant | n | ρ(attention) | ρ(rollout) | ρ(recency control) |
|---|---|---|---|---|
| full | 5 | 0.225 ± 0.293 | 0.453 ± 0.245 | 0.565 ± 0.217 |
| no_entropy | 3 | 0.285 ± 0.351 | 0.373 ± 0.331 | 0.667 ± 0.227 |
| no_revin | 5 | 0.578 ± 0.215 | 0.456 ± 0.194 | 0.582 ± 0.205 |

**Reading, with the noise caveat above in mind.** With n = 3 and an estimator sd of ~0.18 per run, only a difference well above ~0.3 could be taken seriously. Treat this table as a direction, not a result, until the remaining seeds arrive.

### What does move faithfulness: RevIN, not the regulariser

`full` (RevIN on) and the headline `no_revin` (RevIN off) both have 5 seeds and differ only in RevIN. Paired by seed — identical explained windows and, within a pair, the same permutation stream, so much of the estimator noise cancels in the difference (common random numbers). That is why these paired tests resolve effects smaller than the ~0.18 unpaired sd would suggest.

| Metric | RevIN on (full) | RevIN off (headline) | Δ | paired t p | on > off |
|---|---|---|---|---|---|
| fidelity, attention-ranked | 0.335 ± 0.246 | 0.040 ± 0.230 | +0.296 | 0.004 | 5/5 |
| Spearman(attention, SHAP) | 0.839 ± 0.061 | 0.484 ± 0.335 | +0.355 | 0.072 | 5/5 |
| fidelity, SHAP-ranked | 0.449 ± 0.190 | 0.401 ± 0.189 | +0.048 | 0.047 | 5/5 |
| fidelity, permutation-ranked | 0.455 ± 0.183 | 0.382 ± 0.178 | +0.073 | 0.015 | 5/5 |

**The configuration selected for accuracy is the one with the least faithful variable attention.** Switching RevIN off — which the validation loss chose, and which gives the paper its first place by MAE — takes attention-ranked fidelity from 0.34 to 0.04 (p = 0.004, 5 of 5 seeds) and the attention–SHAP agreement from 0.84 to 0.48 (5 of 5 seeds). This is an accuracy–faithfulness trade-off, and it is the most defensible XAI finding in this analysis: it is paired, on equal budgets, and consistent across every seed.

On the time axis the picture is different and weaker: neither configuration's temporal attention beats the recency control in [occlusion_time.md](analysis/occlusion_time.md), so there is no faithfulness on that axis to trade.

### Verdict for P0-3 and P0-4

1. The 5-seed-vs-1-seed comparison is gone: every model now has 5 seeds under identical conditions, and SHAP exists for all 11.
2. Our model is **not** the most faithful by SHAP or permutation fidelity: Crossformer is level (Δ −0.02, p = 0.27). It beats 9 of 10 baselines on average, but only TimesNet and Autoformer survive Holm on Jena. The published claim of a clear lead rested on a frozen RNG stream.
3. The built-in attention of the headline model is not faithful (0.04, indistinguishable from zero).
4. The entropy regulariser has **no detectable effect** on faithfulness on either axis in the clean `full` vs `no_entropy` comparison. Its claimed benefit should be dropped or reworded as sparsity only.
5. What does matter is RevIN: turning it off bought accuracy and cost attention faithfulness. That trade-off is the honest XAI story for the resubmission.

### Structural issue with the ablation table

Every ablation in [src/train.py](src/train.py) is defined relative to `full`, which has RevIN **on**. The headline model is `no_revin`. So each ablation row measures what a component contributes to a model the paper does not report. The clean fix is to re-base the ablations on the headline configuration (`use_revin=False` plus the component removed). That is a GPU-track change and would replace P1-2 as currently specified; it is flagged here, not done.

And, as recorded in [results_hygiene.md](analysis/results_hygiene.md), `no_var_attn` disables the entropy term too — with `use_var_attn=False` the attention is a constant and `var_entropy` has no gradient path — so that row removes two things at once.
