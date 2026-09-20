# Resubmission checklist, against the reviews themselves

Built from `reviews/reviewer_1.md` and `reviews/reviewer_2.md` (the reviews are
kept out of git). Every item of both reviewers appears below with a status, the
file that holds the result and the numbers that go into the paper. Numbers are
copied from the generated files, none are typed from memory. The reviewers'
own wording is quoted under each item, so this checklist still works after a
fresh clone, where `reviews/` is absent.

Reviewer 1 numbers his items only in *Specific Revisions Required*; the Major
and Minor lists are numbered here in the order he wrote them, which is the
order those references point to.

Status legend: **done** — computed, result in the repo · **manuscript** — a
text/table edit in the paper, no computation needed · **partly** — computed but
something is still missing · **open** — not done.

Headline configuration: `XAI-MeteoFormer`, ablation `no_revin`, seq_len 96,
5 seeds, HuberLoss(delta=1) — unchanged by this round.

## Data state

| What | Value | Where |
|---|---|---|
| published runs | **300 rows = 300 checkpoints = 300 predictions**, one-to-one by tag | `analysis/results_clean.csv` |
| training-recipe variants (MSE, outage augmentation) | 230 rows, kept out of the published table | `analysis/results_variants.csv` |
| window sweep (seq_len is not part of the run key) | 24/48/192 × 2 datasets × 5 seeds | `analysis/results_seqlen{L}.csv`, `checkpoints/seqlen{L}/` |
| repo commits the GPU runs used | `e887d9a`, `761f816`, `12630a3` | Kaggle logs in `outputs/kaggle/` |

---

## Reviewer 1 — Major

### Major #1 — "a post-hoc SHAP ranking on Crossformer beats your built-in attention" — **done, and he is right**
> *"The central interpretability thesis is refuted by the authors' own Table 5. … Table 5 shows that a post-hoc SHAP ranking computed on the competitor Crossformer achieves a higher fidelity gain on Jena (0.518) than XAI-MeteoFormer's own attention (0.463); on Beijing the proposed model (0.436, 5-seed mean) merely edges Crossformer's single-seed 0.352. A post-hoc method applied to another model therefore matches or beats the 'inherently interpretable' one."*
- **Computed:** fidelity for all 11 models × 5 seeds on both datasets under identical conditions, plus a deterministic estimator (`analysis/fidelity_deterministic.py`) that removes the Monte-Carlo noise of the permutation metric.
- **Files:** `analysis/xai_fidelity_v2.md`, `paper/tables/fidelity_*.tex`, `analysis/fidelity_det.csv`.
- **Numbers (deterministic metric, primary; `analysis/fidelity_det.csv`, 5 seeds):** SHAP-ranked fidelity of the **headline** model is **5th of 11 on both datasets**. Jena — Crossformer 0.259 ± 0.028, iTransformer 0.241 ± 0.006, PatchTST 0.233 ± 0.004, TimesNet 0.209 ± 0.019, **ours 0.207 ± 0.007**. Beijing — Crossformer 0.169 ± 0.016, PatchTST 0.158 ± 0.011, DLinear 0.144 ± 0.001, iTransformer 0.142 ± 0.003, **ours 0.130 ± 0.017**; Crossformer's advantage there is significant (Δ −0.040, Holm p = 0.019). The **built-in attention** is at **0.012 ± 0.065** (Jena) and **0.046 ± 0.092** (Beijing) — effectively zero, now with a tight interval.
- **Why the earlier ranking was wrong:** under the permutation metric our model looked 2nd, but its seed sd was ≈ 0.19, i.e. the ordering was noise. The deterministic estimator drops our sd to 0.007, and the honest position is 5th.
- **What the paper must say:** the reviewer is right, and the final numbers are worse than his: the built-in attention explains nothing, and even our SHAP explanation ranks 5th of 11. The interpretability claim has to be dropped entirely, and the contribution restated as accuracy (first by MAE, significant) plus a set of externally validated negative results about attention.

### Major #2 — abstract contradicts itself on Spearman agreement — **manuscript**
- No computation involved. The corrected agreement numbers are in `analysis/xai_fidelity_v2.md` (ρ per model, 5 seeds). The "0.436 … the lowest of all eleven models" sentence has to go, and the missing Table 6 either added or its reference removed.

### Major #3 — temporal interpretability unvalidated; asks for a time-axis occlusion test — **done**
> *"Temporal interpretability is entirely unvalidated. … the temporal-attention branch (which past hours matter) is validated only by self-consistency with attention rollout — another attention-derived, internally computed quantity, not an independent method. … Either validate temporal importance with a perturbation/occlusion test along the time axis (mask patches, measure MAE delta, compare to the a/rollout ranking) or retract the temporal-explanation claim."*
- **Computed:** patch occlusion along the time axis with a recency control; block gaps as an independent second check.
- **Files:** `analysis/occlusion_time.md`, `analysis/block_missing.md`.
- **Numbers:** occluding the last patch (t = 80–96 h) costs **+2.039 MAE**; all ten earlier patches together 0.241. Temporal attention spans 0.0905–0.0924 — flat. ρ(attention) = +0.578 vs **ρ(recency control) = +0.582**, paired difference −0.0036, p = 0.95; rollout is worse (+0.456).
- **What the paper must say:** the temporal-explanation claim is retracted. The defensible finding is that the model is a last-16-hours model — confirmed externally, and shared by all 11 models (`block_missing.md`).

### Major #4 — fidelity is a 5-seed mean against 1-seed baselines — **done**
> *"The fidelity comparison is statistically invalid (5-seed mean vs 1-seed). … A mean cannot be compared to a single realization … Recompute all baselines' SHAP fidelity over five seeds (and report SD) before any claim is made."*
- **Computed:** 110 runs (11 models × 5 seeds × 2 datasets), same 2 560 windows, same attribution budget, same perturbation grid, paired RNG; SHAP now exists for every model (`analysis/xai_patches.py` works around the in-place normalisation that broke input gradients).
- **Files:** `analysis/xai_fidelity_v2.md`, `paper/tables/fidelity_*.tex`.
- **Numbers:** the permutation estimator's own noise on a single fixed checkpoint is sd ≈ 0.17–0.18 — as large as the spread across trained seeds, so it could not separate models closer than ≈ 0.2, and the published sd of 0.037 was an artefact of reusing one RNG stream. It is therefore replaced as the primary metric by a deterministic one (channel replaced by its own window mean, fixed reference orderings, verified bit-identical on re-runs); the permutation version moves to the appendix (`paper/tables/fidelity_perm_appendix_*.tex`).

### Major #5 — no significance testing — **done**
> *"No statistical significance testing despite noise-level margins. … Beijing MAE 4.227 +/- 0.073 vs Crossformer 4.247 +/- 0.095, a 0.02 gap against standard deviations of 0.07-0.10. These are noise-level differences presented as victories. … Either demonstrate significance with proper tests or stop claiming superiority where none exists."*
- **Files:** `analysis/significance.md`, `analysis/significance_dm.csv`, `paper/tables/significance_*.tex`. Diebold–Mariano on the per-window loss with a Newey–West HAC variance and the Harvey–Leybourne–Newbold correction, Holm within each family; paired t and Wilcoxon over seed means alongside, with the note that n = 5 cannot reach p < 0.05 for Wilcoxon.
- **Numbers:** on absolute loss ours is significantly better than **all 10 baselines on both datasets**. On squared loss **Crossformer is significantly better than ours on both datasets**; on Beijing five further baselines are not significantly different.
- **What the paper must say:** superiority claims are restricted to MAE, where they are significant, and the squared-loss result for Crossformer is stated plainly.

### Major #6 — RevIN effect unexplained; asks for a window-length sweep on both datasets — **done**
> *"The 'RevIN discovery' is unexplained and over-promoted. … A result flagged as central but left unexamined is not a contribution; it is an uncontrolled anomaly. At minimum, sweep the input-window length (24/48/96/192 h) and both datasets to locate the cause, or drop the claim that RevIN 'hurts.'"*
- **Files:** `analysis/window_sweep.md`; mechanism in `analysis/occlusion_time.md` and `analysis/block_missing.md`.
- **Numbers** (headline model, 5 seeds per length, DM against L = 96 on the common test windows):

  | L (h) | Jena val loss | Jena MAE | Beijing val loss | Beijing MAE |
  |---|---|---|---|---|
  | 24 | **0.1389** | **2.940 ± 0.020** | **0.1580** | **4.058 ± 0.042** |
  | 48 | 0.1417 | 2.994 ± 0.021 | 0.1590 | 4.107 ± 0.090 |
  | 96 (published) | 0.1437 | 3.025 ± 0.031 | 0.1592 | 4.151 ± 0.039 |
  | 192 | 0.1457 | 3.041 ± 0.023 | 0.1648 | 4.234 ± 0.059 |

  Shorter is better, monotonically, on both datasets; every difference against L = 96 is significant by DM except Beijing L = 192 on squared loss.
- **Decision:** the headline stays at 96. Switching would require re-running all baselines at 24 and re-opens the tuning asymmetry this round closes. The sweep is reported as the requested localisation of the effect: the 96 h context is largely unused, exactly as the occlusion profile predicts.
- **Not run:** the mean-only RevIN variant (`revin_mean_only` exists in `src/train.py`). Authors' decision — the mechanism is confirmed three times over (occlusion, block gaps, window sweep), so #6 is answered by explanation rather than by a fourth normalisation variant.

### Major #7 — the entropy penalty may manufacture the agreement it "validates" — **done**
> *"The entropy penalty may manufacture the very agreement it 'validates.' … sparsity is imposed, not discovered … The no_entropy ablation reports only accuracy, never fidelity or Spearman agreement … Report fidelity and agreement for the no_entropy model to show the regularizer actually earns its interpretability claim."*
- **Computed:** `no_entropy` extended to 5 seeds on both datasets, plus a clean control on the headline configuration (`no_revin` vs `no_revin+no_entropy`, 5 seeds × 2 datasets) — the earlier comparison confounded the regulariser with RevIN.
- **Files:** `analysis/xai_fidelity_v2.md` (P0-4), `analysis/ablation_norevin.md`, `analysis/occlusion_time.md`.
- **Numbers (accuracy):** removing the entropy term from the headline changes MAE by **+0.027** on Jena (DM ΔL1 p_Holm = 2.6e-05) and **−0.042** on Beijing (p_Holm = 0.030) — on Beijing the model is *better* without it.
- **Numbers (faithfulness, deterministic metric):** the regulariser does **not** buy faithfulness — after Holm nothing is significant and the direction runs against it. What does move faithfulness is RevIN: on Jena the configuration selected for accuracy (`no_revin`) has attention-ranked fidelity **0.012 ± 0.065** against **0.181 ± 0.079** for `full` (paired p = 0.001, 5 of 5 seeds). Both findings survived the change of metric, which is the best evidence that they are real.

## Reviewer 1 — Minor

| # | Item | Status |
|---|---|---|
| 1 | `R2(%)` label — R² is dimensionless | **manuscript** |
| 2 | Section 4.4 says 0.491, Table 5 says 0.436 | **manuscript**; the recomputed values in `xai_fidelity_v2.md` supersede both |
| 3 | Frost F1 = 0.412 called "optimal" while AUC is the lowest | **manuscript**; current numbers in `paper/tables/events_*.tex` (Jena: ours AUC 0.975 vs Crossformer 0.980, F1 0.701 vs 0.717) and `analysis/frost_events.md`, which documents the baseline adaptation and the validation-selected threshold |
| 4 | Novelty is a recombination — say so honestly | **manuscript**; supported by `analysis/tuning_budget.md` |
| 5 | Eq. (15)/(16) rollout notation garbled | **manuscript** |
| 6 | *"Only one Beijing station (Aotizhongxin) is used; cross-station generalization is untested despite 12 stations being available in PRSA."* | **partly** — zero-shot transfer to the other 11 stations is running (`analysis/cross_station.py`), 204 of 660 runs done; the in-domain control reproduces `results_clean.csv` exactly |
| 7 | *"The promised missing-data robustness sweep (5/10/20% synthetic missingness) is absent, yet operational decision support is the paper's stated motivation."* | **done** — `analysis/missing_robustness.md`: first on both datasets at 20 % scattered missingness; `analysis/block_missing.md` adds the block-outage case, where it is not |
| 8 | *"The GPU/CPU configuration is 'not recorded'; for reproducibility, at least hardware and framework/library versions should be reported (the code link is implied but not provided)."* | **done** — `analysis/reproducibility.md`, `analysis/environment.json`, `analysis/requirements_frozen.txt`: Tesla T4, Python 3.12.13, torch 2.10.0+cu128, CUDA 12.8, TSLib pinned at `4e938a17` |
| 9 | "First attempt to benchmark attention against SHAP…" overstated | **manuscript** |

---

## Reviewer 2

### #1 — attention is computed after the transformations, and correlated predictors make the selection unstable — **done**
> *"The proposed variable attention is applied after convolutional and transformer feature transformations and therefore reflects importance within the final attention mechanism rather than necessarily the contribution of the original meteorological variables to the forecast. … Entropy regularization may encourage sparse attention, but sparsity does not by itself establish faithfulness and may result in unstable or arbitrary selection among correlated variables."*
- **File:** `analysis/attention_stability.md`.
- **Numbers:** Jena, mean CV of the within-group winner's weight **0.96**, of the group total **0.62** — grouping correlated channels reduces the instability only ~1.5×. The attention concentrates on `hour_cos`/`hour_sin`, not on physical drivers; on Beijing `hour_cos` alone takes 0.44–0.54 of the mass for RH, P and WS.
- **What the paper must say:** the per-variable physical-driver claim is not supported; report the instability as a finding and move the explanation claims onto SHAP.

### #2 — symmetric explainability evaluation (seeds, samples, budgets) and an external temporal test — **done**
> *"The comparison with SHAP is asymmetric because the proposed model is evaluated over multiple seeds whereas some baselines appear to rely on a single seed. A fair comparison should use the same number of seeds, evaluation samples, perturbation conditions and attribution budgets for all methods."*
- Same evidence as R1 Major #3 and #4: `analysis/xai_fidelity_v2.md` (11 models × 5 seeds, identical budgets), `analysis/occlusion_time.md` (external time-axis test), `analysis/temporal_pooling_target_specificity.md`.
- **On "temporal pooling is not clearly target-specific":** correct — it is per input channel and **shared across the four targets**, while the variable attention is per target. The paper must say "per-variable temporal attention, shared across targets".

### #3 — per-target and per-horizon reporting; frost-head protocol; F1/AUC contradiction — **done**
> *"Because the targets have different units and scales, aggregate MAE and RMSE values can obscure important differences between variables. Per-target and per-horizon results should be reported … The frost-classification experiment likewise requires clarification regarding how regression baselines were adapted for classification, whether identical auxiliary heads and training procedures were used, how class imbalance was handled and how F1 thresholds were selected."*
- **Files:** `analysis/per_target_horizon.md`, `paper/tables/per_target_*.tex`, `per_horizon_*.tex`, `analysis/frost_events.md`, `paper/tables/events_*.tex`.
- **Numbers:** first place is **not** uniform across targets. Jena — 1st on temperature (MAE and RMSE), 2nd on RH, P and WS behind Crossformer. Beijing — 1st on T, 4th on RH, 2nd on P and WS. By horizon — 1st at 6/12/24 h on both datasets, 2nd–3rd at 1 h behind DLinear.
- **Frost protocol:** every baseline is scored from its predicted temperature (`-T_pred` as the ranking score, `T_pred <= tau` as the decision); our dedicated head is evaluated separately in the ablation, so two different amounts of supervision are never compared. The threshold is selected on validation. The conclusion's "best F1 and AUC" claim must be corrected: ours is 4th by AUC on Jena and 2nd on Beijing.

### Reporting and consistency paragraph (unnumbered)
| Item | Status |
|---|---|
| Baseline count mismatch | **manuscript** — 10 baselines + our model, as in `paper/tables/main_*.tex` |
| Contradictory SHAP-agreement statements | **manuscript**, superseded by `xai_fidelity_v2.md` |
| Table 5 vs text on Beijing fidelity; missing Table 6 | **manuscript** |
| Matched tuning budget insufficiently documented | **done** — `analysis/tuning_budget.md`: there was **no hyperparameter search for any model**, one trial each, identical architecture-level settings; the asymmetries that do exist (Crossformer/TimesNet/TFT at reduced width, PatchTST given our patch settings) are listed. Crossformer's width is now tested directly in `analysis/crossformer_fullwidth.md` |
| Missing-value treatment, split dates, feature construction, code availability | **done** — `analysis/reproducibility.md` |
| Significance tests / confidence intervals | **done** — see R1 Major #5 |
| Ablation variability across repeated runs | **done** — every ablation is 5 seeds on both datasets: `paper/tables/ablation.tex` (from `full`), `paper/tables/ablation_norevin.tex` (from the headline), `analysis/ablation_norevin.md` |

---

## What this round adds beyond the reviews

- **Crossformer at full width** (`analysis/crossformer_fullwidth.md`): 128×128 (1.70 M params) 3.107 ± 0.018, 256×512 (7.98 M) 3.184 ± 0.073, 256×1024 (10.61 M) 3.175 ± 0.042, ours (1.89 M) **3.025 ± 0.031**. Width does not help Crossformer, so the reduced-width criticism is answered with data instead of an apology.
- **Baseline normalization symmetry** (`analysis/norm_selection.md`): each baseline now runs in the variant with the lower mean validation loss — the rule that selected `no_revin` for our model. Four switch (Autoformer on both datasets, Transformer on Beijing, LSTM on Jena); all four get stronger and our model stays first by MAE. `main_historical_*.tex` keeps the old configuration for the appendix.
- **Training-recipe variants, each swept over all 11 models** (`analysis/aug_robustness.md`, `analysis/loss_mse.md`). Neither replaces the headline: both are worse on clean validation loss, and choosing one because it wins on test would be selection on test.
  - Station-outage augmentation: Jena 6th → **1st** under the outage with clean MAE unchanged (3.028 → 3.023); Beijing degradation +60 % → +48 % but the rank stays 8th and clean MAE goes 4.151 → 4.255. Both datasets reported.
  - MSE instead of Huber: a negative result. The RMSE gap does not close (ours 5.267 → 5.214 while Crossformer goes 5.166 → 5.150) and Beijing MAE loses first place (4.151 → 4.224 against Crossformer 4.247 → 4.110). The humidity dispersion moves only 0.951 → 0.927 against an optimum near 0.86, where Crossformer already sits at 0.850.

## Headline numbers for the paper

| | Jena | Beijing |
|---|---|---|
| MeteoFormer MAE | **3.025 ± 0.031** | **4.151 ± 0.039** |
| best baseline MAE | Crossformer 3.107 ± 0.018 | Crossformer 4.247 ± 0.095 |
| MeteoFormer RMSE | 5.267 ± 0.046 | 8.216 ± 0.049 |
| best baseline RMSE | Crossformer **5.166 ± 0.021** | Crossformer **8.002 ± 0.172** |
| MeteoFormer R² | **0.682 ± 0.005** | **0.659 ± 0.002** |
| DM, absolute loss | significantly better than all 10 | significantly better than all 10 |
| DM, squared loss | Crossformer significantly better | Crossformer significantly better |

## Manuscript-only edits — for the response letter and the yellow mark-up

Nothing below needs a computation. Each entry is the reviewer's own sentence,
then the edit. Order follows the reviews.

**1. R1 Major #2 — the abstract contradicts itself.**
> *"The abstract states the model has 'the highest agreement ... in terms of Spearman r (0.837 on Jena and 0.436 on Beijing, the lowest of all eleven models).' A value that is 'the lowest of all eleven models' is, by definition, the WORST agreement — not the highest. This is a logical impossibility within a single sentence and a misleading summary of the Spearman comparison (Table 6 is not even included in the reviewed PDF). The abstract must be rewritten to state the actual comparison honestly, and the 0.436/'lowest' contradiction removed."*
- **Change:** rewrite the abstract sentence. State the agreement values as they are, and drop the word "highest" — on Beijing our agreement is *not* the highest. Recomputed per-model ρ values are in `analysis/xai_fidelity_v2.md`; the abstract must quote those, not the old ones.

**2. R1 Minor #1 — percent sign on R².**
> *"Table 3 labels R2 as 'R2(%)'; R2 is dimensionless and must not carry a percent sign."*
- **Change:** `R²(%)` → `R²` in the Table 3 header and anywhere else the label appears. Values stay as they are (`paper/tables/main_*.tex` already writes plain `$R^2$`).

**3. R1 Minor #2 — 0.491 in the text vs 0.436 in Table 5.**
> *"Section 4.4 text states the Beijing fidelity gain as 'reaching 0.491,' while Table 5 reports 0.436 — an internal inconsistency that must be reconciled."*
- **Change:** both numbers are superseded. Section 4.4 and Table 5 must be rewritten from `analysis/xai_fidelity_v2.md` (11 models × 5 seeds, identical budgets), and the text must quote the same number as the table.

**4. R1 Minor #3 — frost head: F1 called optimal while AUC is the lowest.**
> *"The frost head's F1 = 0.412 is presented as 'optimal,' yet its AUC = 0.921 is the LOWEST of all models. The threshold-dependent nature of F1 should be stated up front, not buried in the Discussion."*
- **Change:** say in the results section that F1 depends on the decision threshold and that the threshold is selected on validation (`analysis/frost_events.md`). Remove "optimal". Current standing: ours is 4th by AUC on Jena, 2nd on Beijing (`paper/tables/events_*.tex`).

**5. R1 Minor #4 — novelty is a recombination.**
> *"Novelty is self-described as 'recombination' of established components (single-head attention, entropy regularization, attention rollout, patching, dilated convolutions, RevIN). The methods contribution is incremental and the paper should state this honestly rather than implying an architectural first."*
- **Change:** state the incremental nature in the contributions list, and move the claim of novelty to what is actually new here: the benchmark protocol and the externally validated negative findings about attention.

**6. R1 Minor #5 — rollout notation garbled.**
> *"Equation (15)/(16) rollout notation is garbled ('Al <- Al * Sum_j A[., j]'); the renormalization step is not clearly specified and should be corrected for reproducibility."*
- **Change:** rewrite equations (15) and (16) with the row-normalisation written out explicitly, matching the implementation in `src/xm_models/`.

**7. R1 Minor #9 — "first attempt" claim in Table 1.**
> *"Related-work Table 1's claim of being 'the first attempt to benchmark attention against SHAP, permutation importance, and perturbation in a single paper' is a strong novelty assertion that is likely overstated; several prior works perform similar multi-method XAI benchmarking."*
- **Change:** delete the "first attempt" wording from Table 1 and the related-work text, or replace it with a comparison that does not claim priority.

**8. R2 — baseline count mismatch, contradictory SHAP statements, Table 5 vs text, missing Table 6.**
> *"These include the mismatch between the claimed number of baselines and those actually listed, contradictory statements concerning SHAP agreement, the discrepancy between the Beijing fidelity values reported in Table 5 and the text and references to a Table 6 that is apparently missing."*
- **Change:** (a) make the baseline count agree everywhere — 10 baselines plus our model, exactly the rows of `paper/tables/main_*.tex`; (b) remove the contradictory SHAP-agreement sentences and quote `xai_fidelity_v2.md` once; (c) reconcile the Beijing fidelity value as in item 3; (d) either include Table 6 (per-model Spearman agreement, available in `xai_fidelity_v2.md`) or delete every reference to it.

**9. R2 #2 / R2 #3 — the target-specific temporal attention claim, and the frost conclusion.**
> *"Temporal pooling is not clearly target-specific and the validation relies largely on agreement with attention rollout, which is another internal attention-based measure rather than an independent attribution or perturbation test."*
> *"In particular, the manuscript contains an apparent contradiction: XAI-MeteoFormer is described as having the best F1 and AUC in the conclusion, whereas the results indicate that it has the lowest AUC."*
- **Change:** (a) replace every "target-specific temporal attention" with "per-variable temporal attention, shared across the four targets" — this is verified in `analysis/temporal_pooling_target_specificity.md`, the pooling is per input channel; (b) delete the "best F1 and AUC" sentence from the conclusion and state the actual ranks.

## Still open

1. **Cross-station transfer** (R1 Minor #6): 456 of 660 runs left; local, resumes from `analysis/cross_station.csv`.
2. **Manuscript edits**: R1 Major #2, R1 Minor #1, #2, #3, #4, #5, #9, and the whole R2 reporting paragraph. None need computation; all need the text rewritten against the numbers above.
3. **If a recipe variant is ever promoted to the headline**, fidelity must be recomputed for it: 10 deterministic + 10 permutation runs, about 1 h locally.

## The two decisive objections, and where we stand

Reviewer 1 rejected on two grounds. Both now have an answer, and one of them is
not the answer the paper originally wanted:

1. *"Not competitive on accuracy — Crossformer beats it on every primary metric
   on Jena."* The comparison he saw was against our `full` configuration
   (MAE 3.195). The headline `no_revin` model is **3.025 vs 3.107**, first by MAE
   on both datasets and significant against all ten baselines. Crossformer keeps
   RMSE, at every width, and that is now stated instead of glossed.
2. *"Built-in attention is not a useful explanation."* Confirmed, and the final
   numbers are worse than the ones he saw: on the deterministic metric the
   attention's fidelity is 0.012 (Jena) and 0.046 (Beijing), and even our SHAP
   explanation is 5th of 11 on both datasets. The attention is also flat on the
   time axis and unstable on the variable axis. The paper's XAI
   claim must be rebuilt around what survived: an externally validated statement
   about *where* the model looks (the last 16 h), SHAP as the explanation of
   record, and the accuracy–faithfulness trade-off that RevIN exposes.
