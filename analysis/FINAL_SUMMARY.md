# Resubmission checklist, item by item

Status as of 2026-09-19, after the Kaggle GPU track closed. One line per
reviewer item: what was computed, where the result lives, which numbers go
into the paper. Numbers are copied from the files named. None are typed from memory.

> **The full reviewer text is not in this repository.** The item list
> below was rebuilt from the reviewer items named in the working
> sessions and from the item numbers cited in `analysis/*.md`. Items marked
> *inferred* were never named explicitly. **R1 Major #1, #2, #5, R1 Minor
> #1–#5 and every other R2 item are unknown here.** Check this list against
> the reviews before you write the response letter.

Headline configuration throughout: `XAI-MeteoFormer`, ablation `no_revin`
(RevIN off), seq_len 96, 5 seeds.

## Data state

| What | Value | Where |
|---|---|---|
| runs in the main table source | **300 rows = 300 checkpoints = 300 predictions**, one-to-one by tag | `analysis/results_clean.csv`, `checkpoints/`, `predictions/` |
| new since the last submission | 17 (P0-1b/c, P0-4) + 85 (P1-1, P1-2) + 60 (ablations from no_revin) | `results_new_kaggle*.csv` |
| window sweep (not in the main table: seq_len is not in the run key) | L = 24/48/192 × 2 datasets × 5 seeds, 60 runs | `analysis/results_seqlen{L}.csv`, `checkpoints/seqlen{L}/`, `predictions/seqlen{L}/` |
| Kaggle logs, per kernel | all runs rc=0 | `outputs/kaggle/<kernel>/logs/` |
| repo commit the Kaggle runs used | `761f816` (ablations from no_revin, sweep seeds 3–4), `e887d9a` (all others). `src/` is identical for the runs that predate `761f816` | — |

---

## Reviewer 1

### Major #3 — external validation of the temporal axis *(inferred)*
- **Computed:** patch occlusion with a recency control; block gaps as a second external check (P0-5).
- **Files:** `analysis/occlusion_time.md`, `analysis/block_missing.md`.
- **Numbers for the paper:** occluding the last patch (80–96 h) costs **+2.039 MAE**, all ten earlier patches together cost 0.241. Temporal attention spans 0.0905–0.0924, which is flat. ρ(attention) = +0.578 vs ρ(recency) = +0.582, p = 0.95. Block gaps: losing the last 16 h hurts every model, and ours the most (Jena +73 % vs +0.6 % for the oldest 16 h; Beijing +60 % vs +0.4 %). **First place does not survive an outage in the last 16 h.**
- **Status:** done. The claim that temporal attention explains *when* the model looks must be dropped.

### Major #4 — explanation fidelity on equal budgets for all 11 models (P0-3)
- **Computed:** fidelity for 11 models × 5 seeds on both datasets, GradientSHAP for every model, same windows, paired RNG.
- **Files:** `analysis/xai_fidelity_v2.md`, `paper/tables/fidelity_*.tex`.
- **Numbers for the paper:** the estimator's own noise is sd ≈ 0.17–0.18 on a single fixed checkpoint, the same size as the spread across seeds. At this budget, models closer than ~0.2 cannot be separated. The published sd of 0.037 for our model was an artefact of reusing one RNG stream.
- **Status:** done. Drop the old 5-seeds-vs-1-seed comparison.

### Major #6 — where the RevIN effect comes from: window sweep + mean-only RevIN (P0-2)
- **Computed:** seq_len sweep 24/48/192 vs the published 96, headline model, both datasets.
- **Files:** `analysis/window_sweep.md` (Welch t-test on seeds, plus Diebold–Mariano on the test windows common to both lengths).
- **Numbers for the paper** (5 seeds for every L; MAE mean ± sd; DM against L = 96 on the common test windows, Holm):

  | L (h) | Jena val loss | Jena MAE | Jena DM ΔL1 / ΔL2 | Beijing val loss | Beijing MAE | Beijing DM ΔL1 / ΔL2 |
  |---|---|---|---|---|---|---|
  | 24 | **0.1389** | **2.940 ± 0.020** | −0.086\* / −1.234\* | **0.1580** | **4.058 ± 0.042** | −0.117\* / −3.376\* |
  | 48 | 0.1417 | 2.994 ± 0.021 | −0.032\* / −0.652\* | 0.1590 | 4.107 ± 0.090 | −0.050\* / −2.341\* |
  | 96 (published) | 0.1437 | 3.025 ± 0.031 | — | 0.1592 | 4.151 ± 0.039 | — |
  | 192 | 0.1457 | 3.041 ± 0.023 | +0.029\* / +0.447\* | 0.1648 | 4.234 ± 0.059 | +0.067\* / +1.906 (n.s.) |

  Shorter is better on both datasets, and monotonically so. **Validation also selects L = 24 on both datasets**, so this is a validation-based choice, not a test-set one. This fits occlusion: the model uses the last ~16 h.
- **Side check (single tests, raw p, not in any table):** on the published test windows, the L = 24 model vs Crossformer 128x128 (L = 96) gives Jena ΔL1 −0.168 (p < 1e-15) and ΔL2 −0.183 (p = 0.34); Beijing ΔL1 −0.213 (p = 4.4e-16) and ΔL2 +0.068 (p = 0.94). **At L = 24 Crossformer's significant squared-loss advantage disappears.** Caveat: every baseline ran at L = 96. Switching our headline to L = 24 without a window sweep for the baselines would reopen the fairness question.
- **Status:** done, **without the mean-only variant**. The authors decided not to run it: the RevIN mechanism is now confirmed three times over — occlusion (`occlusion_time.md`), block gaps (`block_missing.md`) and this window sweep — and a fourth normalization variant would add a data point, not an explanation. Major #6 is answered by the mechanism, and the answer is stated as such. `revin_mean_only` remains available in `src/train.py` if a reviewer insists.

### Major #7 — separate the entropy regulariser from real faithfulness (P0-4)
- **Computed:** `no_entropy` is now at 5 seeds on both datasets (it was 3 seeds, Jena only). **New:** a clean control on the headline configuration, `no_revin` vs `no_revin+no_entropy`, 5 seeds × 2 datasets.
- **Files:** `analysis/xai_fidelity_v2.md` (P0-4 section, `full` vs `no_entropy`), `analysis/ablation_norevin.md` (accuracy of `no_revin+no_entropy`), `analysis/occlusion_time.md`.
- **Numbers for the paper (accuracy, from `ablation_norevin.md`):** removing the entropy term from the headline gives ΔMAE **+0.027** on Jena (DM ΔL1 p_Holm = 2.6e-05, ΔL2 n.s.) and **−0.042** on Beijing (DM ΔL1 p_Holm = 0.030). On Beijing the model is *better* without it.
- **Numbers for the paper (faithfulness):** the P0-4 table in `xai_fidelity_v2.md` is still **provisional** (n = 3, `full` vs `no_entropy`, Jena). The strongest XAI finding there is paired and uses 5 seeds: RevIN, not the regulariser, drives faithfulness (attention-ranked fidelity 0.335 → 0.040 when RevIN is off, p = 0.004, 5/5 seeds).
- **Missing:** the fidelity and occlusion analysis has **not been re-run on the new checkpoints** (`no_entropy` s3–4 and Beijing; `no_revin+no_entropy`). Until it is, P0-4 on faithfulness stays provisional. That work belongs to the analysis session (xai_fidelity_v2.py), locally, without GPU quota.

### Minor #6 — cross-station transfer
- **Computed (in progress):** zero-shot transfer of the Aotizhongxin checkpoints to the other 11 PRSA stations — no retraining, no refitting, the Aotizhongxin training scaler applied to every station, same channels and same test months. 11 models x 5 seeds x 12 stations.
- **File:** `analysis/cross_station.py` -> `analysis/cross_station.csv`, `analysis/cross_station.md`, `paper/tables/cross_station.tex`.
- **Status:** **204 of 660 runs done**, paused while the analysis session held the GPU; it resumes from the CSV. The in-domain control reproduces `results_clean.csv` exactly (Autoformer seed 0: 5.188077 both sides), so the harness is validated.

### Minor #7 — robustness to missing inputs
- **Files:** `analysis/missing_robustness.md` (scattered dropouts), `analysis/block_missing.md` (block outage).
- **Numbers for the paper:** at 20 % scattered missing values our model **stays first on both datasets**. Lower ranks reshuffle: ranks 3–9 on Jena, ranks 2–3 on Beijing. A block outage of the last 16 h removes our first place (see Major #3).
- **Status:** done.

### Minor #8 — reproducibility
- **Files:** `analysis/reproducibility.md`, `analysis/environment.json`, `analysis/requirements_frozen.txt`.
- **Numbers:** TSLib pinned to `4e938a1767106324dd753b2a44832bf870a0252e` and verified on Kaggle. Kaggle environment: Python 3.12.13, torch 2.10.0+cu128, CUDA 12.8, Tesla T4.
- **Status:** done.

---

## Reviewer 2

### #1 — attention stability across correlated predictors
- **File:** `analysis/attention_stability.md`.
- **Numbers:** Jena, mean CV of the within-group winner's weight **0.96**, of the group total **0.62**. Attention concentrates on `hour_cos`/`hour_sin`, not on physical drivers.
- **Status:** done. The claim about per-variable physical drivers must be weakened.

### #2 — is the temporal attention target-specific?
- **File:** `analysis/temporal_pooling_target_specificity.md`.
- **Answer:** **no.** Temporal pooling is per input channel and shared across the four targets. Variable attention is per target.
- **Status:** done. The paper's wording must change.

### (no number) per-target / per-horizon metrics, frost events
- **Files:** `analysis/per_target_horizon.md`, `analysis/frost_events.md`, `paper/tables/per_target_*.tex`, `per_horizon_*.tex`, `events_*.tex`.
- **Status:** done. These tables were **not changed** by this pass (verified against the snapshot byte for byte).

### (no number) tuning budget and fairness
- **Files:** `analysis/tuning_budget.md`, `analysis/tuning_symmetry_audit.md`, `analysis/crossformer_fullwidth.md` (P0-1), `analysis/norm_selection.md` (P1-1).
- **P0-1, Crossformer at full width** (Jena, 5 seeds, batch 64, T4):

  | Configuration | params | MAE | RMSE | R² |
  |---|---|---|---|---|
  | Crossformer 128x128 (published) | 1.70 M | 3.107 ± 0.018 | 5.166 ± 0.021 | 0.678 ± 0.004 |
  | Crossformer 256x512 | 7.98 M | 3.184 ± 0.073 | 5.241 ± 0.063 | 0.675 ± 0.004 |
  | Crossformer 256x1024 | 10.61 M | 3.175 ± 0.042 | 5.237 ± 0.039 | 0.672 ± 0.005 |
  | MeteoFormer (ours) | 1.89 M | 3.025 ± 0.031 | 5.267 ± 0.046 | 0.682 ± 0.005 |

  A wider Crossformer is **not** better. DM ΔL1 favours ours at every width (p_Holm ≤ 1e-11). **On squared loss the published 128x128 Crossformer beats ours** (ΔL2 +1.05, p = 4.9e-07). At 256 width that gap is not significant (p = 0.27).
- **P1-1, normalization symmetry:** under the pre-registered validation rule, 4 of 10 (model, dataset) pairs switch variant: Autoformer → `on` on both datasets, Transformer → `on` on Beijing, LSTM → `off` on Jena. Against the **selected** variants, ours is significantly better on ΔL1 for all five baselines on both datasets. On ΔL2 it is significantly better on Jena for all five. On Beijing it is significantly better against Transformer, Informer and Autoformer, but **not** against DLinear (p = 0.18) or LSTM (ΔL2 +1.84, n.s.).
- **Needs an update:** `analysis/tuning_budget.md` still says "138 runs" and that Crossformer was under-provisioned. Both are superseded by P0-1 (300 runs; the width does not help Crossformer).

### (no number) comparable budgets / seeds
- **Status:** done. Every main configuration, every ablation (from `full` and from `no_revin`), the normalization variants, the window sweep and the fidelity runs now have **5 seeds**.

---

## Cross-cutting

### Significance (replaces the unpaired Welch test)
- **Files:** `analysis/significance.md`, `analysis/significance_dm.csv`, `paper/tables/significance_*.tex`.
- **Unchanged by this pass.** Byte-identical: no new runs of the headline or of the published baselines.
- **Numbers:** on absolute loss ours is significantly better than all 10 baselines on both datasets. On squared loss, **Crossformer is significantly better than ours on both datasets**. On Beijing, DLinear, LSTM, PatchTST, TimesNet and iTransformer are n.s.

### Ablation
- **Headline table, new:** `paper/tables/ablation_norevin.tex`, `analysis/ablation_norevin.md`. Components are removed from `no_revin`, 5 seeds, both datasets, with DM.
  - Jena: removing the fusion gate (+0.067, DM L1 and L2 significant) or variable attention together with the entropy term (+0.075, L1 and L2 significant) hurts. Multi-scale and temporal attention are n.s.
  - Beijing: only the fusion gate helps (+0.044, L1 significant). Removing the entropy term (−0.042) or the event head (−0.043, L1 and L2 significant) **improves** MAE.
  - The variable-attention row removes **both** variable attention and the entropy term. It is labelled that way in both files.
- **Appendix table:** `paper/tables/ablation.tex` (from `full`, RevIN on, Jena), now 5 seeds for every row. Changed numbers are listed below.

### Training-recipe variants (both swept over all 11 models)
Two recipes were trained on the GPU track, each for every model, 5 seeds, both datasets, so neither is a private advantage for the proposed model. **Neither replaces the headline**: the selection rule is the lower mean validation loss, and on clean validation data both are worse for our model.

**Station-outage augmentation** — `analysis/aug_robustness.md`, `paper/tables/aug_robustness.tex`. During training, with probability 0.5 one contiguous 16 h block of the window is blanked and refilled by interpolation.

| | validation loss | clean MAE | outage MAE | degradation | rank under outage |
|---|---|---|---|---|---|
| Jena, published | 0.1437 | 3.028 | 5.251 | +73% | 6 / 11 |
| **Jena, augmented** | 0.1448 | **3.023** | **4.616** | **+53%** | **1 / 11** |
| Beijing, published | 0.1592 | 4.150 | 6.637 | +60% | 8 / 11 |
| Beijing, augmented | 0.1605 | 4.255 | 6.307 | +48% | 8 / 11 |

On Jena the augmentation removes the weakness outright at no cost to clean accuracy. On Beijing it reduces the damage but does not change the standing, and clean MAE gets worse (4.151 -> 4.255, still first). **Both datasets are reported**; Jena alone would be selective reporting.

**MSE instead of Huber** — `analysis/loss_mse.md`, `paper/tables/loss_mse_appendix.tex`. A negative result, kept in the appendix. It does not close the RMSE gap (Jena 5.267 -> 5.214 while Crossformer goes 5.166 -> 5.150) and **costs us first place on MAE on Beijing** (4.151 -> 4.224 against Crossformer 4.247 -> 4.110). The mechanism is real but too weak: the dispersion of our humidity forecast (sd of prediction over sd of observation) moves only 0.951 -> 0.927 against an optimum near 0.86, where Crossformer already sits at 0.850.

### Changed numbers in `paper/tables/`
- `ablation.tex`: every row went from 3 seeds to 5. MAE: no multiscale 3.122 -> 3.128, no cls 3.132 -> 3.127, no fusion 3.211 -> 3.206, no entropy 3.186 -> 3.178, no temp attn 3.194 -> 3.199, no var attn 3.198 -> 3.183.
- `main_{jena,beijing}.tex`: now the validation-selected normalization. Four rows changed — Autoformer (both datasets) and Transformer (Beijing) move to `on`, LSTM (Jena) to `off`; all four baselines get **stronger**, and our model stays first by MAE on both datasets. The historical version is `main_historical_*.tex`.
- `per_target_*`, `per_horizon_*`, `events_*`: two rows each, the switched baselines.
- `significance_*.tex`: recomputed against the selected variants. **No conclusion changed**: significantly better than all 10 baselines on absolute loss, on both datasets; Crossformer still significantly better on squared loss. `analysis/significance_historical*` keeps the previous comparison and is byte-identical to it.
- New: `ablation_norevin.tex`, `aug_robustness.tex`, `loss_mse_appendix.tex`, `cross_station.tex` (pending the remaining runs).
- Unchanged: `fidelity_*.tex` — owned by `analysis/xai_fidelity_v2.py` (the analysis session is replacing the estimator with a deterministic one); `report.py` no longer overwrites them from the superseded single-seed metrics.

## Decisions taken

1. **Window length: 96 stays.** Validation picks L = 24 on both datasets and test agrees, but switching would require the baselines at L = 24 as well, which re-opens the asymmetry this round is closing. The sweep is reported as an analysis of context redundancy, consistent with occlusion and the block gaps.
2. **Main table: validation-selected normalization**, for all 11 models, historical in the appendix. Every downstream analysis was re-run for the four switched variants.
3. **Mean-only RevIN: not run** (see Major #6).
4. **Cross-station transfer: being run** (Minor #6).
5. **Headline model unchanged.** Neither the augmented nor the MSE recipe replaces it; both are worse on clean validation loss, and picking one because it wins on test is selection on test.

## Still open

- **The reviewer text.** This list was rebuilt from the items named in the working sessions and the numbers cited in `analysis/*.md`; R1 Major #1, #2, #5, R1 Minor #1–#5 and several R2 items are unknown here. Check it against the reviews before writing the response letter.
- **Cross-station transfer**: 456 of 660 runs left, local, resumes from `analysis/cross_station.csv`.
- **P0-4 faithfulness**: the fidelity estimator is being replaced by a deterministic one in the analysis session; the P0-4 section of `xai_fidelity_v2.md` is provisional until that lands.
- **`analysis/tuning_budget.md`** still says 138 runs and calls Crossformer under-provisioned; both are superseded (300 runs; width does not help Crossformer).
- **If the augmented or MSE recipe is ever promoted** to the headline, fidelity has to be recomputed for it: 10 runs deterministic plus 10 permutation, about 1 h locally.
