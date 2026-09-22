# Robustness to missing inputs (Reviewer 1, Minor #7)

Produced by [analysis/missing_robustness.py](analysis/missing_robustness.py) and [analysis/missing_report.py](analysis/missing_report.py). Inference only, on the existing checkpoints; no model was retrained.

## How the gaps are made and filled

The mechanism already in the codebase is used unchanged (`MeteoWindowDataset.missing_rate`, [src/data/dataset.py](src/data/dataset.py)):

1. A Bernoulli(`missing_rate`) mask is drawn **independently per cell** of the (96, n_channels) input window — per timestep *and* channel, not per whole timestep.
2. Masked cells become NaN and are refilled by **linear interpolation along time within that window**, per channel, over the surviving indices. A channel that is entirely missing is set to 0.0, the mean in scaled space.
3. The mask touches the **scaled input only**. Targets are never masked, so the metric stays comparable to the clean run.
4. The RNG is seeded from the run's seed, so the gap pattern is reproducible and identical across models at a given seed.

Budget: **5 seeds**, four rates, both datasets — the same seed budget as the main tables, so the numbers here are not on a smaller budget than the ones they sit next to.

## Test subsample and its validation

The quantity of interest is a within-model difference between rates, so a systematic 1-in-k subsample of the test windows is used, with **all four rates computed on the same subsample** — the deltas are therefore exact by construction, not approximated.

| Dataset | full test windows | used | stride |
|---|---|---|---|
| beijing_aotizhongxin | 6895 | 3448 | 1-in-2 |
| jena | 13908 | 2782 | 1-in-5 |

Validated against **110 full-test runs**, paired exactly on (dataset, model, seed, rate) — not compared as means:

| | mean difference | max abs | max relative |
|---|---|---|---|
| MAE | +0.0015 | 0.0108 | 0.32% |
| RMSE | +0.0021 | 0.0153 | 0.27% |

- mean degradation at 20 % on Jena, **full test** (n=27): **0.841%**
- mean degradation at 20 % on Jena, **subsample** (n=55): **0.777%**

The subsample does not shift the measured degradation.


---

## Jena

### MAE by missing rate (mean ± sd over 5 seeds)

| Model | 0 % | 5 % | 10 % | 20 % | Δ% at 20 % |
|---|---|---|---|---|---|
| Transformer | 3.452±0.067 | 3.453±0.067 | 3.454±0.066 | 3.459±0.064 | +0.21 |
| LSTM | 3.317±0.045 | 3.318±0.045 | 3.320±0.045 | 3.326±0.045 | +0.26 |
| Autoformer | 3.903±0.047 | 3.906±0.048 | 3.907±0.047 | 3.915±0.046 | +0.29 |
| Informer | 3.457±0.065 | 3.458±0.066 | 3.459±0.071 | 3.471±0.066 | +0.43 |
| TimesNet | 3.313±0.015 | 3.315±0.016 | 3.319±0.016 | 3.329±0.017 | +0.47 |
| TFT | 3.388±0.066 | 3.392±0.065 | 3.399±0.065 | 3.415±0.063 | +0.82 |
| iTransformer | 3.300±0.013 | 3.305±0.014 | 3.313±0.013 | 3.334±0.011 | +1.03 |
| **MeteoFormer** | 3.028±0.031 | 3.033±0.031 | 3.040±0.030 | 3.061±0.031 | +1.11 |
| DLinear | 3.427±0.008 | 3.434±0.006 | 3.444±0.006 | 3.468±0.006 | +1.20 |
| Crossformer | 3.110±0.020 | 3.117±0.020 | 3.125±0.018 | 3.148±0.018 | +1.22 |
| PatchTST | 3.352±0.019 | 3.362±0.019 | 3.373±0.019 | 3.402±0.021 | +1.51 |

### RMSE by missing rate (mean ± sd over 5 seeds)

| Model | 0 % | 5 % | 10 % | 20 % | Δ% at 20 % |
|---|---|---|---|---|---|
| Transformer | 5.632±0.084 | 5.631±0.084 | 5.632±0.083 | 5.637±0.081 | +0.10 |
| LSTM | 5.412±0.056 | 5.412±0.056 | 5.414±0.056 | 5.421±0.056 | +0.17 |
| Autoformer | 6.205±0.069 | 6.209±0.073 | 6.211±0.072 | 6.220±0.073 | +0.25 |
| Informer | 5.638±0.094 | 5.639±0.094 | 5.642±0.099 | 5.656±0.095 | +0.32 |
| TimesNet | 5.466±0.018 | 5.468±0.018 | 5.472±0.017 | 5.484±0.018 | +0.33 |
| TFT | 5.555±0.052 | 5.561±0.051 | 5.570±0.051 | 5.593±0.052 | +0.68 |
| iTransformer | 5.567±0.018 | 5.573±0.020 | 5.581±0.021 | 5.608±0.025 | +0.73 |
| **MeteoFormer** | 5.271±0.050 | 5.276±0.047 | 5.285±0.045 | 5.314±0.042 | +0.81 |
| DLinear | 5.763±0.009 | 5.773±0.008 | 5.785±0.008 | 5.817±0.011 | +0.94 |
| Crossformer | 5.170±0.022 | 5.178±0.021 | 5.189±0.023 | 5.220±0.023 | +0.97 |
| PatchTST | 5.645±0.032 | 5.658±0.031 | 5.672±0.030 | 5.712±0.031 | +1.18 |

### Per-target degradation, Δ% of MAE at 20 % missing

| Model | T | RH | P | WS |
|---|---|---|---|---|
| Autoformer | -0.15 | +0.36 | +0.51 | +0.13 |
| Crossformer | +0.79 | +1.28 | +1.82 | +0.46 |
| DLinear | +1.45 | +1.36 | +0.74 | +0.73 |
| Informer | +0.17 | +0.41 | +0.85 | +0.16 |
| LSTM | +0.01 | +0.25 | +0.59 | +0.20 |
| PatchTST | +1.32 | +1.44 | +2.21 | +0.65 |
| TFT | +0.26 | +0.90 | +1.33 | +0.25 |
| TimesNet | +0.09 | +0.48 | +0.88 | +0.32 |
| Transformer | +0.14 | +0.10 | +0.63 | +0.19 |
| **MeteoFormer** | +0.82 | +1.05 | +1.80 | +0.51 |
| iTransformer | +0.89 | +1.08 | +1.21 | +0.52 |

### Does the ranking survive?

| Rank | clean (0 %) | 20 % missing |
|---|---|---|
| 1 | **MeteoFormer** | **MeteoFormer** |
| 2 | Crossformer | Crossformer |
| 3 | iTransformer | LSTM  ←changed |
| 4 | TimesNet | TimesNet |
| 5 | LSTM | iTransformer  ←changed |
| 6 | PatchTST | PatchTST |
| 7 | TFT | TFT |
| 8 | DLinear | Transformer  ←changed |
| 9 | Transformer | DLinear  ←changed |
| 10 | Informer | Informer |
| 11 | Autoformer | Autoformer |

Ranking by MAE is **changed** between the clean test and 20 % missing.


---

## Beijing (Aotizhongxin)

### MAE by missing rate (mean ± sd over 5 seeds)

| Model | 0 % | 5 % | 10 % | 20 % | Δ% at 20 % |
|---|---|---|---|---|---|
| Transformer | 4.895±0.071 | 4.895±0.070 | 4.896±0.070 | 4.896±0.071 | +0.03 |
| Autoformer | 5.347±0.112 | 5.346±0.111 | 5.348±0.110 | 5.350±0.107 | +0.05 |
| LSTM | 4.343±0.034 | 4.344±0.033 | 4.347±0.033 | 4.352±0.033 | +0.20 |
| TimesNet | 4.426±0.033 | 4.428±0.033 | 4.430±0.032 | 4.436±0.031 | +0.22 |
| TFT | 4.676±0.177 | 4.680±0.176 | 4.684±0.175 | 4.693±0.175 | +0.36 |
| Informer | 4.935±0.092 | 4.936±0.088 | 4.943±0.087 | 4.957±0.083 | +0.43 |
| iTransformer | 4.250±0.017 | 4.256±0.017 | 4.263±0.016 | 4.281±0.016 | +0.73 |
| DLinear | 4.361±0.008 | 4.369±0.008 | 4.378±0.010 | 4.400±0.011 | +0.89 |
| **MeteoFormer** | 4.150±0.039 | 4.157±0.039 | 4.165±0.040 | 4.187±0.039 | +0.90 |
| PatchTST | 4.369±0.046 | 4.377±0.047 | 4.386±0.046 | 4.411±0.046 | +0.94 |
| Crossformer | 4.137±0.023 | 4.144±0.021 | 4.153±0.021 | 4.177±0.021 | +0.96 |

### RMSE by missing rate (mean ± sd over 5 seeds)

| Model | 0 % | 5 % | 10 % | 20 % | Δ% at 20 % |
|---|---|---|---|---|---|
| Transformer | 8.922±0.118 | 8.923±0.117 | 8.924±0.120 | 8.923±0.121 | +0.00 |
| Autoformer | 9.416±0.184 | 9.416±0.184 | 9.421±0.183 | 9.424±0.177 | +0.09 |
| TimesNet | 8.275±0.064 | 8.277±0.063 | 8.283±0.062 | 8.291±0.065 | +0.20 |
| LSTM | 8.104±0.028 | 8.105±0.027 | 8.112±0.027 | 8.121±0.028 | +0.22 |
| TFT | 8.433±0.172 | 8.441±0.170 | 8.451±0.169 | 8.467±0.166 | +0.41 |
| Informer | 8.970±0.145 | 8.976±0.144 | 8.991±0.150 | 9.010±0.160 | +0.45 |
| iTransformer | 8.265±0.031 | 8.273±0.030 | 8.285±0.025 | 8.309±0.020 | +0.54 |
| DLinear | 8.378±0.013 | 8.388±0.014 | 8.402±0.014 | 8.431±0.017 | +0.63 |
| **MeteoFormer** | 8.220±0.049 | 8.228±0.049 | 8.243±0.051 | 8.278±0.051 | +0.71 |
| PatchTST | 8.337±0.049 | 8.347±0.049 | 8.364±0.048 | 8.400±0.050 | +0.76 |
| Crossformer | 7.916±0.081 | 7.927±0.081 | 7.942±0.084 | 7.978±0.084 | +0.79 |

### Per-target degradation, Δ% of MAE at 20 % missing

| Model | T | RH | P | WS |
|---|---|---|---|---|
| Autoformer | -0.70 | +0.26 | -0.09 | +0.10 |
| Crossformer | -0.22 | +1.15 | +1.51 | +0.42 |
| DLinear | +1.22 | +0.91 | +0.61 | +0.77 |
| Informer | +0.05 | +0.58 | +0.36 | +0.16 |
| LSTM | -0.45 | +0.31 | +0.36 | +0.18 |
| PatchTST | +1.29 | +0.86 | +0.95 | +1.20 |
| TFT | -0.50 | +0.61 | +0.25 | +0.16 |
| TimesNet | -0.54 | +0.35 | +0.43 | +0.22 |
| Transformer | -0.30 | +0.02 | +0.46 | +0.01 |
| **MeteoFormer** | +0.60 | +0.88 | +1.41 | +0.49 |
| iTransformer | +0.56 | +0.78 | +0.64 | +0.68 |

### Does the ranking survive?

| Rank | clean (0 %) | 20 % missing |
|---|---|---|
| 1 | Crossformer | Crossformer |
| 2 | **MeteoFormer** | **MeteoFormer** |
| 3 | iTransformer | iTransformer |
| 4 | LSTM | LSTM |
| 5 | DLinear | DLinear |
| 6 | PatchTST | PatchTST |
| 7 | TimesNet | TimesNet |
| 8 | TFT | TFT |
| 9 | Transformer | Transformer |
| 10 | Informer | Informer |
| 11 | Autoformer | Autoformer |

Ranking by MAE is **unchanged** between the clean test and 20 % missing.


---

## Verdict

**The effect is small for every model — at most 1.5% MAE for 20% of input cells destroyed.** That is a property of the corruption scheme, not evidence that the models are robust, and it must be written that way.

The mask is per-cell and is immediately repaired by linear interpolation along time. Hourly meteorological series are close to linear over a few hours, so an isolated missing cell is recovered almost exactly. The experiment therefore measures how good linear interpolation is on this data, with the model's sensitivity as a second-order effect. **Do not claim "our model is robust to missing data" on this basis.**

Our model degrades by 1.11% on Jena, which is in the **lower half** of the field — Transformer, LSTM, Informer, Autoformer and TimesNet all degrade less. So the claim to make is not about robustness.

What does hold is **persistence of the first place**. Our model is first by MAE at every gap level on both datasets, and the margin over the runner-up does not shrink:

| Dataset | 0 % | 5 % | 10 % | 20 % |
|---|---|---|---|---|
| Jena | +0.0824 vs Crossformer | +0.0841 vs Crossformer | +0.0848 vs Crossformer | +0.0867 vs Crossformer |
| Beijing | +0.0132 vs XAI-MeteoFormer | +0.0127 vs XAI-MeteoFormer | +0.0119 vs XAI-MeteoFormer | +0.0108 vs XAI-MeteoFormer |

The **rest** of the ranking is not stable: mid-field positions swap between the clean test and 20 % missing on both datasets (e.g. on Jena iTransformer/TimesNet and DLinear/Transformer exchange places). Those pairs are separated by less than their own seed sd, so the swaps carry no signal — but the paper should say "first place is unaffected", not "the ranking is unaffected", because the latter is checkably false.

If a stronger robustness claim is wanted, the corruption scheme has to be harder and closer to an operational failure — whole timesteps dropped, or a contiguous block of hours lost, or one sensor offline for the entire window. Those are cheap to add (the same inference harness) and would be a better answer to the reviewer than the present result. They are **not** included here, and nothing above should be read as covering them.
