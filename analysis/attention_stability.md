# Stability of the variable attention among correlated predictors

Reviewer 2 #1. Produced by [analysis/attention_stability.py](analysis/attention_stability.py) from `xai/*.npz`; no model is run.

Channels are grouped into connected components of the |Pearson r| ≥ 0.9 graph on the **training** split. Within such a group the channels are nearly interchangeable, so which one a softmax attention selects is close to arbitrary. The question is whether the *group's* total weight is stable even when the winner inside it is not.


---

## Jena

5 seeds. Correlation groups found (|r| ≥ 0.9 on the training split):

- **G1**: T, Tpot, Tdew, VPmax, VPact, SH, H2OC, rho (|r| from 0.804 to 1.000)
- **G2**: WS, WSmax (|r| from 0.971 to 0.971)

### Per-channel weight across seeds (mean±sd), top channels

| Target | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| T | hour_cos 0.275±0.178 | hour_sin 0.195±0.089 | rho 0.130±0.232 | doy_cos 0.057±0.036 | wy 0.042±0.027 |
| RH | hour_cos 0.270±0.132 | hour_sin 0.198±0.064 | rho 0.126±0.227 | wx 0.047±0.006 | P 0.046±0.008 |
| P | rho 0.165±0.245 | hour_cos 0.105±0.058 | doy_cos 0.105±0.066 | hour_sin 0.097±0.056 | Tdew 0.056±0.061 |
| WS | hour_cos 0.246±0.140 | hour_sin 0.181±0.055 | rho 0.113±0.198 | P 0.053±0.009 | wy 0.045±0.017 |

### Within-group instability vs group-level stability

`winner flips` = number of distinct arg-max channels inside the group across the 5 seeds (1 = always the same channel, 5 = a different one every seed). `CV` = sd/mean.

| Target | Group | winner flips | modal winner (freq) | winner weight mean±sd (CV) | **group total mean±sd (CV)** |
|---|---|---|---|---|---|
| T | G1 | 3 | rho (3/5) | 0.153±0.222 (1.45) | **0.288±0.245 (0.85)** |
| T | G2 | 2 | WS (4/5) | 0.013±0.009 (0.66) | **0.022±0.014 (0.63)** |
| RH | G1 | 2 | rho (4/5) | 0.147±0.218 (1.48) | **0.273±0.221 (0.81)** |
| RH | G2 | 2 | WS (4/5) | 0.015±0.008 (0.52) | **0.025±0.013 (0.50)** |
| P | G1 | 2 | rho (4/5) | 0.195±0.229 (1.18) | **0.464±0.208 (0.45)** |
| P | G2 | 2 | WSmax (4/5) | 0.023±0.012 (0.53) | **0.038±0.018 (0.48)** |
| WS | G1 | 3 | rho (3/5) | 0.131±0.190 (1.45) | **0.271±0.197 (0.73)** |
| WS | G2 | 1 | WS (5/5) | 0.021±0.009 (0.42) | **0.036±0.019 (0.53)** |

**Summary — Jena:** mean CV of the within-group winner's weight = **0.961**; mean CV of the group total = **0.622** (ratio 1.5×).

### Seed-to-seed variability of the single top channel

| Target | top channel | CV across seeds |
|---|---|---|
| T | hour_cos | 0.65 |
| RH | hour_cos | 0.49 |
| P | rho | 1.48 |
| WS | hour_cos | 0.57 |


---

## Beijing (Aotizhongxin)

5 seeds. Correlation groups found (|r| ≥ 0.9 on the training split):

_none — no channel pair reaches the threshold_


### Per-channel weight across seeds (mean±sd), top channels

| Target | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| T | doy_cos 0.137±0.076 | CO 0.088±0.026 | SO2 0.087±0.021 | NO2 0.085±0.027 | PM10 0.074±0.022 |
| RH | hour_cos 0.435±0.349 | T 0.170±0.126 | hour_sin 0.108±0.203 | Tdew 0.083±0.095 | P 0.048±0.037 |
| P | hour_cos 0.489±0.387 | hour_sin 0.111±0.190 | T 0.099±0.108 | doy_cos 0.071±0.109 | Tdew 0.054±0.071 |
| WS | hour_cos 0.539±0.397 | hour_sin 0.136±0.259 | T 0.126±0.144 | Tdew 0.050±0.074 | doy_cos 0.045±0.082 |

### Seed-to-seed variability of the single top channel

| Target | top channel | CV across seeds |
|---|---|---|
| T | doy_cos | 0.56 |
| RH | hour_cos | 0.80 |
| P | hour_cos | 0.79 |
| WS | hour_cos | 0.74 |


---

## Verdict

**The variable attention is not stable across seeds, and grouping correlated channels does not rescue it.**

- Jena: mean CV of the within-group winner **0.96**, of the group total **0.62**. Grouping reduces the variability by about 1.5×, but a CV of 0.62 still means the group weight moves by roughly 62% of its own size from seed to seed.

Two further observations that a reviewer will make before we do:

1. **The attention concentrates on the time-of-day encodings, not on physical predictors.** On Jena `hour_cos` and `hour_sin` take the top two slots for T, RH and WS; on Beijing `hour_cos` alone takes 0.44-0.54 of the mass for RH, P and WS. An explanation whose headline finding is "the model uses the hour of day" is not wrong, but it is not the physical-driver story the paper currently tells.
2. **The sd is of the same order as the mean for the top channels** (e.g. Beijing WS: `hour_cos` 0.539±0.397). Reporting a mean attention vector over seeds without its sd, as the current figures do, hides this entirely.

### What this means for the resubmission

The claim that the variable attention identifies *which physical variable* drives a target cannot be supported by this evidence. Three honest options, in descending order of strength:

- **Report the instability as a finding.** Attention over near-collinear meteorological inputs is unstable by construction, quantify it, and present the group-level weight as the only quantity worth reading. This is a real contribution and it is what the numbers support.
- **Keep the attention as an internal mechanism** and move the explanation claims onto permutation/SHAP importance, which is model-agnostic and comparable across the baselines.
- Drop the per-channel attention figures and keep only the group-level and fidelity results.

Note on Beijing: no channel pair reaches |r| ≥ 0.9, so the group analysis is vacuous there. That is itself notable — RH on Beijing is *derived* from T and Tdew via the Magnus formula (src/data/prepare.py), so the three are functionally dependent even though their linear correlation stays below the threshold.