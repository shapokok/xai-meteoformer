# Dispersion calibration fitted on validation (appendix)

Produced by [analysis/variance_calibration.py](analysis/variance_calibration.py). Every prediction is rescaled around its own validation mean, `pred -> m + a (pred - m)`, with **`a` fitted per (model, dataset, seed, channel) on the validation split, then frozen and applied to test**. Test is scored once, after freezing; no test quantity enters the fit. The same protocol is applied to **all 11 models**.

This is an analysis of the mechanism found in [analysis/error_decomposition.md](analysis/error_decomposition.md), not a change of result: the headline model, `paper/tables/main_*.tex` and every other table stay as they are.

**The question:** the test oracle wanted a ≈ 0.86 on humidity for our model. Can validation find it?

## Jena

### Is the coefficient reachable on validation? (humidity)

| Model | a fitted on validation | a from the test oracle | difference | sd ratio on test |
|---|---|---|---|---|
| **MeteoFormer** | 0.868 ± 0.032 | 0.861 | +0.007 | 0.951 |
| Autoformer | 0.873 ± 0.019 | 0.877 | -0.004 | 0.865 |
| Crossformer | 0.973 ± 0.036 | 0.972 | +0.001 | 0.850 |
| DLinear | 0.925 ± 0.005 | 0.921 | +0.004 | 0.842 |
| Informer | 0.853 ± 0.024 | 0.834 | +0.019 | 0.970 |
| LSTM | 0.945 ± 0.022 | 0.931 | +0.014 | 0.868 |
| PatchTST | 0.963 ± 0.021 | 0.976 | -0.013 | 0.798 |
| TFT | 0.919 ± 0.029 | 0.934 | -0.015 | 0.858 |
| TimesNet | 0.913 ± 0.009 | 0.927 | -0.014 | 0.865 |
| Transformer | 0.859 ± 0.025 | 0.838 | +0.021 | 0.964 |
| iTransformer | 0.888 ± 0.016 | 0.895 | -0.007 | 0.888 |

### What the frozen coefficient does to the test error

| Model | MAE before | MAE after | RMSE before | RMSE after | MSE RH before | MSE RH after |
|---|---|---|---|---|---|---|
| **MeteoFormer** | 3.025 | 3.019 | 5.267 | 5.162 | 93.15 | 88.58 |
| Autoformer | 3.904 | 3.864 | 6.203 | 6.087 | 116.09 | 113.50 |
| Crossformer | 3.107 | 3.103 | 5.166 | 5.153 | 85.63 | 85.18 |
| DLinear | 3.423 | 3.435 | 5.758 | 5.735 | 106.73 | 105.84 |
| Informer | 3.452 | 3.316 | 5.631 | 5.403 | 99.11 | 92.29 |
| LSTM | 3.317 | 3.298 | 5.410 | 5.371 | 92.91 | 91.88 |
| PatchTST | 3.348 | 3.345 | 5.635 | 5.611 | 104.57 | 104.40 |
| TFT | 3.384 | 3.371 | 5.551 | 5.502 | 98.24 | 97.35 |
| TimesNet | 3.310 | 3.293 | 5.464 | 5.412 | 95.68 | 94.73 |
| Transformer | 3.449 | 3.364 | 5.627 | 5.435 | 99.36 | 92.88 |
| iTransformer | 3.295 | 3.264 | 5.561 | 5.456 | 100.44 | 97.90 |

Our RMSE 5.267 → 5.162; Crossformer 5.166 → 5.153. Our MAE 3.025 → 3.019.

## Beijing (Aotizhongxin)

### Is the coefficient reachable on validation? (humidity)

| Model | a fitted on validation | a from the test oracle | difference | sd ratio on test |
|---|---|---|---|---|
| **MeteoFormer** | 0.845 ± 0.019 | 0.819 | +0.025 | 0.933 |
| Autoformer | 0.736 ± 0.043 | 0.821 | -0.085 | 0.839 |
| Crossformer | 0.939 ± 0.056 | 0.916 | +0.023 | 0.849 |
| DLinear | 0.840 ± 0.010 | 0.912 | -0.071 | 0.814 |
| Informer | 0.817 ± 0.026 | 0.780 | +0.036 | 0.952 |
| LSTM | 0.820 ± 0.016 | 0.885 | -0.064 | 0.870 |
| PatchTST | 0.858 ± 0.018 | 0.931 | -0.073 | 0.797 |
| TFT | 0.872 ± 0.032 | 0.908 | -0.036 | 0.828 |
| TimesNet | 0.825 ± 0.015 | 0.872 | -0.046 | 0.871 |
| Transformer | 0.743 ± 0.033 | 0.792 | -0.049 | 0.923 |
| iTransformer | 0.783 ± 0.011 | 0.847 | -0.064 | 0.896 |

### What the frozen coefficient does to the test error

| Model | MAE before | MAE after | RMSE before | RMSE after | MSE RH before | MSE RH after |
|---|---|---|---|---|---|---|
| **MeteoFormer** | 4.151 | 4.157 | 8.216 | 8.057 | 252.42 | 242.40 |
| Autoformer | 5.347 | 5.582 | 9.417 | 9.579 | 311.08 | 326.61 |
| Crossformer | 4.247 | 4.319 | 8.002 | 7.931 | 230.74 | 225.14 |
| DLinear | 4.362 | 4.540 | 8.381 | 8.394 | 257.87 | 258.71 |
| Informer | 4.932 | 4.913 | 8.965 | 8.617 | 283.82 | 261.02 |
| LSTM | 4.343 | 4.512 | 8.103 | 8.219 | 237.74 | 247.69 |
| PatchTST | 4.370 | 4.523 | 8.336 | 8.410 | 256.28 | 261.94 |
| TFT | 4.676 | 4.791 | 8.431 | 8.502 | 252.59 | 259.41 |
| TimesNet | 4.426 | 4.558 | 8.273 | 8.301 | 247.20 | 251.21 |
| Transformer | 4.895 | 5.020 | 8.923 | 8.867 | 285.88 | 285.12 |
| iTransformer | 4.251 | 4.440 | 8.266 | 8.272 | 250.76 | 251.73 |

Our RMSE 8.216 → 8.057; Crossformer 8.002 → 7.931. Our MAE 4.151 → 4.157.

## Does the squared-loss deficit survive calibration?

Diebold–Mariano on the per-window loss, our model against Crossformer, **both calibrated with their own validation-fitted coefficients**; per-window loss averaged over seeds, HAC variance, HLN correction. Negative favours our model.

| Dataset | ΔL1 before | ΔL1 after | ΔL2 before | ΔL2 after |
|---|---|---|---|---|
| Jena | -0.0824 (9.93e-12) | -0.0845 (8.22e-15) | +1.0513 (1.63e-07) | +0.0934 (0.607) |
| Beijing (Aotizhongxin) | -0.0964 (0.000492) | -0.1620 (3.48e-06) | +3.4437 (0.000365) | +2.0195 (0.0759) |

## Reading

- **Jena:** validation puts the humidity coefficient at **0.868 ± 0.032** against the test oracle's 0.861 (difference +0.007); applying it changes our RMSE by -0.105 and our MAE by -0.006.
- **Beijing (Aotizhongxin):** validation puts the humidity coefficient at **0.845 ± 0.019** against the test oracle's 0.819 (difference +0.025); applying it changes our RMSE by -0.159 and our MAE by +0.007.

**Answer to the question.** The coefficient is reachable: validation finds 0.868 on Jena against the test oracle's 0.861, a difference of 0.007 — inside the seed-to-seed spread. The two routes do **not** hit the same wall: training with MSE moved the dispersion only from 0.951 to 0.927 ([analysis/loss_mse.md](analysis/loss_mse.md)), while a coefficient fitted on validation reaches the optimum the test oracle wanted. The over-dispersion is therefore not a property of the Huber criterion; it is a scale the model does not learn but that a single validation-fitted number recovers.

What it does not do is change the paper: calibration is applied to every model, our MAE lead survives it and the squared-loss comparison with Crossformer is in the table above. The headline stays uncalibrated.
