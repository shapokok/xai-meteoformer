# Zero-shot transfer across Beijing stations (Reviewer 1, Minor #6)

Produced by [analysis/cross_station.py](analysis/cross_station.py). Every model is trained on **Aotizhongxin only** and applied without retraining to the other 11 PRSA stations. Inputs are scaled with the Aotizhongxin training scaler; nothing is fitted on the target station. Same channels, same chronological split, same test months at every station. 5 seeds per model; baselines in the normalization variant of the main table.

## Control: Aotizhongxin through the same code path

| Model | MAE here | MAE in results_clean | |Δ| |
|---|---|---|---|
| Autoformer | 5.3472 | 5.3472 | 9.5e-07 |
| Crossformer | 4.1377 | 4.1377 | 2.9e-07 |
| DLinear | 4.3623 | 4.3623 | 0.0e+00 |
| Informer | 4.9353 | 4.9320 | 3.3e-03 |
| LSTM | 4.3427 | 4.3427 | 4.8e-07 |
| PatchTST | 4.3703 | 4.3703 | 9.5e-08 |
| TFT | 4.6758 | 4.6758 | 1.9e-07 |
| TimesNet | 4.4257 | 4.4260 | 2.1e-04 |
| Transformer | 4.8946 | 4.8946 | 9.5e-08 |
| MeteoFormer | 4.1505 | 4.1505 | 0.0e+00 |
| iTransformer | 4.2510 | 4.2510 | 1.9e-07 |

Largest deviation 3.3e-03. Every model reproduces its published in-domain MAE. Informer differs by up to 3.3e-03 (<0.3% relative): `ProbAttention` samples keys with `torch.randint` inside the forward pass and `TimesNet` runs an FFT, so neither is bit-reproducible across runs. That is a property of those models, not of this harness.

## Mean over the 11 target stations

Per model: the seed-mean MAE at each station, then mean ± sd across stations; in-domain Aotizhongxin for reference; rank = rank by the cross-station mean MAE.

| Rank | Model | in-domain MAE | transfer MAE (mean ± sd over stations) | transfer RMSE | Δ vs in-domain | stations where 1st |
|---|---|---|---|---|---|---|
| 1 | Crossformer | 4.138 | 4.093 ± 0.089 | 7.792 ± 0.198 | -0.045 | 8/11 |
| 2 | MeteoFormer | 4.151 | 4.114 ± 0.068 | 8.125 ± 0.163 | -0.036 | 3/11 |
| 3 | iTransformer | 4.251 | 4.183 ± 0.096 | 8.109 ± 0.204 | -0.068 | 0/11 |
| 4 | DLinear | 4.362 | 4.281 ± 0.104 | 8.213 ± 0.224 | -0.081 | 0/11 |
| 5 | PatchTST | 4.370 | 4.314 ± 0.106 | 8.203 ± 0.231 | -0.056 | 0/11 |
| 6 | LSTM | 4.343 | 4.315 ± 0.091 | 8.003 ± 0.208 | -0.028 | 0/11 |
| 7 | TimesNet | 4.426 | 4.360 ± 0.091 | 8.122 ± 0.200 | -0.065 | 0/11 |
| 8 | TFT | 4.676 | 4.665 ± 0.081 | 8.375 ± 0.186 | -0.010 | 0/11 |
| 9 | Transformer | 4.895 | 4.842 ± 0.082 | 8.762 ± 0.206 | -0.052 | 0/11 |
| 10 | Informer | 4.935 | 5.015 ± 0.152 | 8.969 ± 0.246 | +0.079 | 0/11 |
| 11 | Autoformer | 5.347 | 5.310 ± 0.069 | 9.338 ± 0.145 | -0.037 | 0/11 |

## Does the ranking survive transfer?

Kendall τ between the in-domain ranking (Aotizhongxin) and the ranking at each target station, by seed-mean MAE; winner per station.

| Station | winner | MeteoFormer rank | Kendall τ vs in-domain |
|---|---|---|---|
| Changping | Crossformer | 3 / 11 | 0.89 |
| Dingling | Crossformer | 3 / 11 | 0.89 |
| Dongsi | Crossformer | 2 / 11 | 1.00 |
| Guanyuan | Crossformer | 2 / 11 | 1.00 |
| Gucheng | MeteoFormer | 1 / 11 | 0.89 |
| Huairou | MeteoFormer | 1 / 11 | 0.82 |
| Nongzhanguan | Crossformer | 2 / 11 | 1.00 |
| Shunyi | Crossformer | 2 / 11 | 0.96 |
| Tiantan | Crossformer | 2 / 11 | 1.00 |
| Wanliu | MeteoFormer | 1 / 11 | 0.93 |
| Wanshouxigong | Crossformer | 2 / 11 | 1.00 |

Kendall τ over stations: mean 0.94, min 0.82, max 1.00.

Per-channel replication: the error budget behind these aggregates is the same in domain and on the unseen stations — our deficit against Crossformer is the humidity channel and nothing else (in domain RH +21.8, transfer +24.1 in MSE units, while temperature stays at -7.8 / -8.2 in our favour). See the replication section of [analysis/error_decomposition.md](analysis/error_decomposition.md).

## MAE per station (seed mean)

| Model | Changping | Dingling | Dongsi | Guanyuan | Gucheng | Huairou | Nongzhanguan | Shunyi | Tiantan | Wanliu | Wanshouxigong |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Crossformer | 3.937 | 3.942 | 4.129 | 4.136 | 4.135 | 4.151 | 4.135 | 3.994 | 4.149 | 4.177 | 4.139 |
| MeteoFormer | 4.010 | 4.017 | 4.176 | 4.168 | 4.111 | 4.084 | 4.178 | 4.036 | 4.178 | 4.121 | 4.179 |
| iTransformer | 4.009 | 4.017 | 4.258 | 4.245 | 4.187 | 4.162 | 4.254 | 4.120 | 4.252 | 4.257 | 4.249 |
| DLinear | 4.104 | 4.104 | 4.362 | 4.362 | 4.301 | 4.228 | 4.362 | 4.199 | 4.362 | 4.342 | 4.362 |
| PatchTST | 4.114 | 4.114 | 4.370 | 4.370 | 4.349 | 4.359 | 4.370 | 4.255 | 4.370 | 4.417 | 4.370 |
| LSTM | 4.178 | 4.161 | 4.330 | 4.335 | 4.353 | 4.470 | 4.340 | 4.224 | 4.349 | 4.380 | 4.341 |
| TimesNet | 4.196 | 4.209 | 4.408 | 4.425 | 4.365 | 4.412 | 4.422 | 4.267 | 4.418 | 4.421 | 4.420 |
| TFT | 4.559 | 4.573 | 4.639 | 4.671 | 4.677 | 4.830 | 4.676 | 4.560 | 4.722 | 4.705 | 4.706 |
| Transformer | 4.674 | 4.750 | 4.841 | 4.865 | 4.849 | 4.942 | 4.859 | 4.765 | 4.939 | 4.884 | 4.898 |
| Informer | 4.996 | 5.085 | 4.966 | 4.943 | 4.995 | 5.445 | 4.942 | 4.876 | 4.948 | 5.003 | 4.962 |
| Autoformer | 5.249 | 5.222 | 5.353 | 5.371 | 5.288 | 5.277 | 5.362 | 5.195 | 5.392 | 5.309 | 5.391 |
