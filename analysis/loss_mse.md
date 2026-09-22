# Training with MSE instead of Huber (appendix, negative result)

Produced by [analysis/training_variants.py](analysis/training_variants.py). Every published run uses `HuberLoss(delta=1)`. Huber linearises the largest residuals, and RMSE lives exactly in that tail, so training with MSE is the direct way to test whether our RMSE gap to Crossformer is a loss-function artefact. **All 11 models** were retrained with MSE, 5 seeds, both datasets.

**Result: it does not close the gap, and on Beijing it costs us first place on MAE.** The variant is not adopted.

## Jena

### MAE

| Model | MAE published | MAE variant | Δ | rank published | rank variant |
|---|---|---|---|---|---|
| **MeteoFormer** | 3.025 | 3.023 | -0.002 | 1 | 1 |
| Crossformer | 3.107 | 3.064 | -0.043 | 2 | 2 |
| LSTM | 3.317 | 3.233 | -0.083 | 5 | 3 |
| TimesNet | 3.310 | 3.270 | -0.040 | 4 | 4 |
| iTransformer | 3.295 | 3.289 | -0.006 | 3 | 5 |
| PatchTST | 3.348 | 3.318 | -0.029 | 6 | 6 |
| TFT | 3.384 | 3.319 | -0.065 | 7 | 7 |
| Transformer | 3.449 | 3.340 | -0.109 | 9 | 8 |
| Informer | 3.452 | 3.341 | -0.111 | 10 | 9 |
| DLinear | 3.423 | 3.443 | +0.020 | 8 | 10 |
| Autoformer | 3.904 | 3.859 | -0.045 | 11 | 11 |

### RMSE

| Model | RMSE published | RMSE variant | Δ | rank published | rank variant |
|---|---|---|---|---|---|
| Crossformer | 5.166 | 5.150 | -0.016 | 1 | 1 |
| **MeteoFormer** | 5.267 | 5.214 | -0.053 | 2 | 2 |
| LSTM | 5.410 | 5.379 | -0.032 | 3 | 3 |
| TimesNet | 5.464 | 5.455 | -0.009 | 4 | 4 |
| TFT | 5.551 | 5.504 | -0.047 | 5 | 5 |
| iTransformer | 5.561 | 5.535 | -0.026 | 6 | 6 |
| Informer | 5.631 | 5.546 | -0.085 | 8 | 7 |
| Transformer | 5.627 | 5.556 | -0.072 | 7 | 8 |
| PatchTST | 5.635 | 5.614 | -0.021 | 9 | 9 |
| DLinear | 5.758 | 5.735 | -0.023 | 10 | 10 |
| Autoformer | 6.203 | 6.177 | -0.027 | 11 | 11 |

### Why it falls short: dispersion of the humidity forecast

`sd ratio RH` = sd of the predicted RH over sd of the observed RH. A squared loss is minimised below 1 (the oracle column, fitted on test, is a measurement of the over-dispersion, not a method).

| Variant | sd ratio RH | oracle a (RH) |
|---|---|---|
| published (Huber) | 0.951 | 0.861 |
| with MSE | 0.927 | 0.884 |
| Crossformer (published) | 0.850 | 0.972 |

## Beijing (Aotizhongxin)

### MAE

| Model | MAE published | MAE variant | Δ | rank published | rank variant |
|---|---|---|---|---|---|
| **MeteoFormer** | 4.151 | 4.224 | +0.073 | 2 | 1 |
| iTransformer | 4.251 | 4.251 | -0.000 | 3 | 2 |
| TimesNet | 4.426 | 4.294 | -0.132 | 7 | 3 |
| LSTM | 4.343 | 4.300 | -0.043 | 4 | 4 |
| PatchTST | 4.370 | 4.347 | -0.023 | 6 | 5 |
| DLinear | 4.362 | 4.416 | +0.054 | 5 | 6 |
| TFT | 4.676 | 4.685 | +0.009 | 8 | 7 |
| Informer | 4.932 | 4.733 | -0.199 | 10 | 8 |
| Transformer | 4.895 | 4.779 | -0.115 | 9 | 9 |
| Autoformer | 5.347 | 5.257 | -0.090 | 11 | 10 |
| Crossformer | 4.138 | _missing_ | | 1 | |

### RMSE

| Model | RMSE published | RMSE variant | Δ | rank published | rank variant |
|---|---|---|---|---|---|
| LSTM | 8.103 | 8.040 | -0.063 | 2 | 1 |
| TimesNet | 8.273 | 8.164 | -0.109 | 5 | 2 |
| **MeteoFormer** | 8.216 | 8.185 | -0.031 | 3 | 3 |
| iTransformer | 8.266 | 8.214 | -0.052 | 4 | 4 |
| PatchTST | 8.336 | 8.340 | +0.004 | 6 | 5 |
| DLinear | 8.381 | 8.369 | -0.012 | 7 | 6 |
| Informer | 8.965 | 8.698 | -0.267 | 10 | 7 |
| TFT | 8.431 | 8.801 | +0.370 | 8 | 8 |
| Transformer | 8.923 | 8.895 | -0.028 | 9 | 9 |
| Autoformer | 9.417 | 9.434 | +0.018 | 11 | 10 |
| Crossformer | 7.916 | _missing_ | | 1 | |

### Why it falls short: dispersion of the humidity forecast

`sd ratio RH` = sd of the predicted RH over sd of the observed RH. A squared loss is minimised below 1 (the oracle column, fitted on test, is a measurement of the over-dispersion, not a method).

| Variant | sd ratio RH | oracle a (RH) |
|---|---|---|
| published (Huber) | 0.933 | 0.819 |
| with MSE | 0.919 | 0.838 |
| Crossformer (published) | 0.849 | 0.914 |

## Reading

The mechanism is real but the effect is too small: MSE moves our RH dispersion only part of the way towards the optimum, while Crossformer is already close to it and improves as well. Per-channel budgets are in `analysis/_tables_mse_{dataset}.md` ([analysis/error_decomposition.py](analysis/error_decomposition.py)).
