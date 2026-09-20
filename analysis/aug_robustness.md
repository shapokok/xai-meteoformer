# Station-outage augmentation (robustness)

Produced by [analysis/training_variants.py](analysis/training_variants.py). During training, with probability 0.5 one contiguous 16 h block of the input window is blanked across every channel and refilled by linear interpolation — the repair an operational forward-fill performs — at a uniformly random position (`block_dropout` in [src/train.py](src/train.py)). **Every one of the 11 models is retrained with it**, 5 seeds, both datasets, so the comparison stays symmetric.

The outage test is the one in [analysis/block_missing.md](analysis/block_missing.md): variant A removes the most recent 16 h of the input window at inference.

**This variant is not the headline model.** The selection rule is the lower mean validation loss, and on clean validation data the augmented model is slightly worse (see the table below). Choosing it because it wins the outage test would be selection on test.

## Jena

### Our model: clean vs outage

| Recipe | validation loss | clean MAE | outage MAE | degradation | rank under outage |
|---|---|---|---|---|---|
| published (Huber, no augmentation) | 0.1437 | 3.028 | 5.251 | +73% | 6 / 11 |
| with outage augmentation | 0.1448 | 3.023 | 4.616 | +53% | 1 / 11 |

### Every model under the outage (MAE, variant A: last 16 h lost)

| Model | published | augmented | Δ | rank published | rank augmented |
|---|---|---|---|---|---|
| **MeteoFormer** | 5.251 | 4.616 | -0.634 | 6 | 1 |
| Crossformer | 4.920 | 4.661 | -0.259 | 1 | 2 |
| TimesNet | 5.005 | 4.796 | -0.209 | 2 | 3 |
| iTransformer | 5.146 | 4.893 | -0.253 | 4 | 4 |
| Autoformer | 5.012 | 4.901 | -0.111 | 3 | 5 |
| LSTM | 5.976 | 4.904 | -1.072 | 9 | 6 |
| PatchTST | 5.230 | 5.092 | -0.138 | 5 | 7 |
| Informer | 5.523 | 5.377 | -0.146 | 7 | 8 |
| TFT | 6.046 | 5.420 | -0.626 | 11 | 9 |
| DLinear | 5.610 | 5.569 | -0.040 | 8 | 10 |
| Transformer | 6.032 | 5.706 | -0.326 | 10 | 11 |

### Accuracy on clean data, every model

| Model | MAE published | MAE variant | Δ | rank published | rank variant |
|---|---|---|---|---|---|
| **MeteoFormer** | 3.025 | 3.022 | -0.003 | 1 | 1 |
| Crossformer | 3.107 | 3.116 | +0.009 | 2 | 2 |
| iTransformer | 3.295 | 3.285 | -0.010 | 3 | 3 |
| LSTM | 3.317 | 3.304 | -0.013 | 5 | 4 |
| PatchTST | 3.348 | 3.323 | -0.025 | 6 | 5 |
| TimesNet | 3.310 | 3.327 | +0.016 | 4 | 6 |
| TFT | 3.384 | 3.374 | -0.010 | 7 | 7 |
| DLinear | 3.423 | 3.418 | -0.005 | 8 | 8 |
| Informer | 3.452 | 3.485 | +0.033 | 10 | 9 |
| Transformer | 3.449 | 3.502 | +0.053 | 9 | 10 |
| Autoformer | 3.904 | 3.874 | -0.030 | 11 | 11 |

## Beijing (Aotizhongxin)

### Our model: clean vs outage

| Recipe | validation loss | clean MAE | outage MAE | degradation | rank under outage |
|---|---|---|---|---|---|
| published (Huber, no augmentation) | 0.1592 | 4.150 | 6.637 | +60% | 8 / 11 |
| with outage augmentation | 0.1605 | 4.255 | 6.307 | +48% | 8 / 11 |

### Every model under the outage (MAE, variant A: last 16 h lost)

| Model | published | augmented | Δ | rank published | rank augmented |
|---|---|---|---|---|---|
| TimesNet | 6.110 | 6.008 | -0.102 | 2 | 1 |
| Autoformer | 6.007 | 6.094 | +0.087 | 1 | 2 |
| Transformer | 6.196 | 6.121 | -0.075 | 3 | 3 |
| iTransformer | 6.249 | 6.128 | -0.121 | 4 | 4 |
| LSTM | 6.893 | 6.128 | -0.765 | 9 | 5 |
| PatchTST | 6.394 | 6.238 | -0.156 | 6 | 6 |
| Crossformer | 6.338 | 6.274 | -0.065 | 5 | 7 |
| **MeteoFormer** | 6.637 | 6.307 | -0.330 | 8 | 8 |
| DLinear | 6.587 | 6.576 | -0.011 | 7 | 9 |
| Informer | 7.033 | 6.783 | -0.251 | 10 | 10 |
| TFT | 8.070 | 7.585 | -0.485 | 11 | 11 |

### Accuracy on clean data, every model

| Model | MAE published | MAE variant | Δ | rank published | rank variant |
|---|---|---|---|---|---|
| **MeteoFormer** | 4.151 | 4.255 | +0.104 | 1 | 1 |
| Crossformer | 4.247 | 4.268 | +0.021 | 2 | 2 |
| iTransformer | 4.251 | 4.275 | +0.024 | 3 | 3 |
| LSTM | 4.343 | 4.329 | -0.014 | 4 | 4 |
| PatchTST | 4.370 | 4.346 | -0.025 | 6 | 5 |
| DLinear | 4.362 | 4.356 | -0.006 | 5 | 6 |
| TimesNet | 4.426 | 4.450 | +0.024 | 7 | 7 |
| TFT | 4.676 | 4.653 | -0.023 | 8 | 8 |
| Informer | 4.932 | 4.864 | -0.068 | 10 | 9 |
| Transformer | 4.895 | 4.892 | -0.003 | 9 | 10 |
| Autoformer | 5.347 | 5.349 | +0.002 | 11 | 11 |

## Reading

- On Jena the augmentation removes the weakness outright: our model goes from 6th to 1st under the outage, and clean accuracy is unchanged.

- On Beijing it helps but does not change the standing: the degradation falls from +60% to +48%, every other model improves too, our rank under the outage stays 8th, and clean MAE gets worse (4.151 → 4.255) while keeping 1st place.

- Both datasets are reported. Showing Jena alone would be selective reporting of the same experiment.
