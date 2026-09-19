# Normalization symmetry for the baselines (P1-1)

Produced by [analysis/norm_selection.py](analysis/norm_selection.py). Selection rule, fixed in advance in [analysis/norm_symmetry_runs.md](analysis/norm_symmetry_runs.md): per (model, dataset) the variant with the lower mean **validation** loss over 5 seeds. `on` = the same RevIN our model can use, added to the baseline; `off` = the baseline as its source defines it (LSTM: without RevIN). Historical = what the published tables use (LSTM `on`, others `off`).

## Jena

| Model | variant | n | val loss | MAE | RMSE | R² | selected |
|---|---|---|---|---|---|---|---|
| DLinear | off | 5 | 0.1733 | 3.423 ± 0.009 | 5.758 ± 0.011 | 0.617 ± 0.001 | **selected**, historical |
| DLinear | on | 5 | 0.1784 | 3.505 ± 0.005 | 5.845 ± 0.004 | 0.602 ± 0.000 |  |
| Transformer | off | 5 | 0.1681 | 3.449 ± 0.068 | 5.627 ± 0.085 | 0.635 ± 0.007 | **selected**, historical |
| Transformer | on | 5 | 0.1759 | 3.579 ± 0.043 | 5.908 ± 0.044 | 0.608 ± 0.006 |  |
| Informer | off | 5 | 0.1684 | 3.452 ± 0.068 | 5.631 ± 0.101 | 0.638 ± 0.009 | **selected**, historical |
| Informer | on | 5 | 0.1715 | 3.487 ± 0.027 | 5.722 ± 0.037 | 0.624 ± 0.003 |  |
| Autoformer | off | 5 | 0.2078 | 3.989 ± 0.113 | 6.269 ± 0.178 | 0.548 ± 0.013 | historical |
| Autoformer | on | 5 | 0.2072 | 3.904 ± 0.047 | 6.203 ± 0.070 | 0.549 ± 0.007 | **selected** |
| LSTM | off | 5 | 0.1615 | 3.317 ± 0.045 | 5.410 ± 0.057 | 0.658 ± 0.004 | **selected** |
| LSTM | on | 5 | 0.1630 | 3.354 ± 0.022 | 5.560 ± 0.017 | 0.639 ± 0.002 | historical |

Selection differs from the historical configuration for: **Autoformer** (→ `on`), **LSTM** (→ `off`).

### Headline model vs the selected variants

| Baseline (selected) | MAE | Δ MAE (ours − baseline) | DM ΔL1 (p_Holm) | DM ΔL2 (p_Holm) |
|---|---|---|---|---|
| DLinear (`off`) | 3.423 | -0.398 | -0.3982\* (<1e-15) | -5.4181\* (<1e-15) |
| Transformer (`off`) | 3.449 | -0.425 | -0.4246\* (<1e-15) | -3.9313\* (<1e-15) |
| Informer (`off`) | 3.452 | -0.427 | -0.4270\* (<1e-15) | -3.9770\* (<1e-15) |
| Autoformer (`on`) | 3.904 | -0.879 | -0.8790\* (<1e-15) | -10.7439\* (<1e-15) |
| LSTM (`off`) | 3.317 | -0.292 | -0.2918\* (<1e-15) | -1.5309\* (1.71e-08) |

Headline MAE: 3.025 ± 0.031. Negative Δ favours the headline model. `*` = p<0.05 after Holm over the five baselines within each loss.

## Beijing (Aotizhongxin)

| Model | variant | n | val loss | MAE | RMSE | R² | selected |
|---|---|---|---|---|---|---|---|
| DLinear | off | 5 | 0.1820 | 4.362 ± 0.008 | 8.381 ± 0.013 | 0.621 ± 0.001 | **selected**, historical |
| DLinear | on | 5 | 0.1836 | 4.440 ± 0.007 | 8.452 ± 0.006 | 0.617 ± 0.001 |  |
| Transformer | off | 5 | 0.2108 | 4.974 ± 0.047 | 8.956 ± 0.170 | 0.597 ± 0.007 | historical |
| Transformer | on | 5 | 0.2023 | 4.895 ± 0.071 | 8.923 ± 0.118 | 0.592 ± 0.004 | **selected** |
| Informer | off | 5 | 0.2017 | 4.932 ± 0.091 | 8.965 ± 0.143 | 0.594 ± 0.003 | **selected**, historical |
| Informer | on | 5 | 0.2051 | 5.033 ± 0.085 | 9.131 ± 0.178 | 0.577 ± 0.013 |  |
| Autoformer | off | 5 | 0.2424 | 5.461 ± 0.035 | 9.551 ± 0.116 | 0.539 ± 0.009 | historical |
| Autoformer | on | 5 | 0.2341 | 5.347 ± 0.112 | 9.417 ± 0.186 | 0.551 ± 0.014 | **selected** |
| LSTM | off | 5 | 0.1879 | 4.685 ± 0.068 | 8.571 ± 0.163 | 0.623 ± 0.006 |  |
| LSTM | on | 5 | 0.1810 | 4.343 ± 0.034 | 8.103 ± 0.027 | 0.638 ± 0.002 | **selected**, historical |

Selection differs from the historical configuration for: **Transformer** (→ `on`), **Autoformer** (→ `on`).

### Headline model vs the selected variants

| Baseline (selected) | MAE | Δ MAE (ours − baseline) | DM ΔL1 (p_Holm) | DM ΔL2 (p_Holm) |
|---|---|---|---|---|
| DLinear (`off`) | 4.362 | -0.212 | -0.2117\* (9.13e-06) | -2.7307 (0.177) |
| Transformer (`on`) | 4.895 | -0.744 | -0.7441\* (<1e-15) | -12.1206\* (2e-15) |
| Informer (`off`) | 4.932 | -0.781 | -0.7815\* (<1e-15) | -12.8863\* (<1e-15) |
| Autoformer (`on`) | 5.347 | -1.197 | -1.1966\* (<1e-15) | -21.1998\* (<1e-15) |
| LSTM (`on`) | 4.343 | -0.192 | -0.1922\* (6.11e-07) | +1.8362 (0.177) |

Headline MAE: 4.151 ± 0.039. Negative Δ favours the headline model. `*` = p<0.05 after Holm over the five baselines within each loss.
