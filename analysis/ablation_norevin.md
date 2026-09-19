# Ablation from the headline configuration (no RevIN)

Produced by [analysis/ablation_norevin.py](analysis/ablation_norevin.py) from [analysis/results_clean.csv](analysis/results_clean.csv). Each row removes one component from the published model (`no_revin`); mean ± sd over seeds. ΔMAE is relative to the headline model: **positive means the component helps**.

The older ablation table, taken from `full` (RevIN on), is a different configuration. It stays in `paper/tables/ablation.tex` as the appendix table on the contribution of each component with RevIN.

> **Row `− variable attention` removes two things.** With `use_var_attn=False` the variable attention is constant, so the entropy regulariser on it has no gradient path. The row is the joint removal of variable attention and the entropy term, not variable attention alone.

## Jena

| Variant | n | MAE | ΔMAE | RMSE | R² | DM ΔL1 (p_Holm) | DM ΔL2 (p_Holm) |
|---|---|---|---|---|---|---|---|
| **MeteoFormer (headline, no RevIN)** | 5 | 3.025 ± 0.031 | — | 5.267 ± 0.046 | 0.682 ± 0.005 | — | — |
| − multi-scale patching | 5 | 3.041 ± 0.006 | +0.016 | 5.259 ± 0.041 | 0.683 ± 0.002 | +0.0158 (0.0632) | -0.0808 (0.617) |
| − variable attention **and entropy term** (joint) | 5 | 3.100 ± 0.079 | +0.075 | 5.330 ± 0.143 | 0.679 ± 0.006 | +0.0748\* (<1e-15) | +0.6814\* (0.000385) |
| − temporal attention | 5 | 3.035 ± 0.047 | +0.011 | 5.254 ± 0.065 | 0.683 ± 0.005 | +0.0106 (0.0632) | -0.1337 (0.464) |
| − fusion gate | 5 | 3.092 ± 0.030 | +0.067 | 5.365 ± 0.036 | 0.678 ± 0.003 | +0.0671\* (7.84e-10) | +1.0423\* (6.66e-05) |
| − entropy term | 5 | 3.052 ± 0.052 | +0.027 | 5.249 ± 0.052 | 0.683 ± 0.003 | +0.0269\* (2.59e-05) | -0.1831 (0.464) |
| − event head | 5 | 3.052 ± 0.037 | +0.027 | 5.332 ± 0.087 | 0.676 ± 0.005 | +0.0270 (0.0502) | +0.6965\* (0.0334) |

DM: ablated minus headline per-window loss, averaged over seeds; **positive Δ means removing the component hurts**. `*` = p<0.05 after Holm over the six variants within this dataset and loss.

## Beijing (Aotizhongxin)

| Variant | n | MAE | ΔMAE | RMSE | R² | DM ΔL1 (p_Holm) | DM ΔL2 (p_Holm) |
|---|---|---|---|---|---|---|---|
| **MeteoFormer (headline, no RevIN)** | 5 | 4.151 ± 0.039 | — | 8.216 ± 0.049 | 0.659 ± 0.002 | — | — |
| − multi-scale patching | 5 | 4.161 ± 0.035 | +0.010 | 8.205 ± 0.082 | 0.658 ± 0.008 | +0.0103 (1) | -0.1715 (1) |
| − variable attention **and entropy term** (joint) | 5 | 4.157 ± 0.132 | +0.006 | 8.255 ± 0.201 | 0.658 ± 0.007 | +0.0063 (1) | +0.6701 (1) |
| − temporal attention | 5 | 4.163 ± 0.052 | +0.012 | 8.147 ± 0.104 | 0.661 ± 0.005 | +0.0121 (0.682) | -1.1147\* (0.00971) |
| − fusion gate | 5 | 4.194 ± 0.042 | +0.044 | 8.131 ± 0.044 | 0.658 ± 0.003 | +0.0436\* (0.0365) | -1.3889 (0.132) |
| − entropy term | 5 | 4.109 ± 0.088 | -0.042 | 8.192 ± 0.153 | 0.662 ± 0.006 | -0.0417\* (0.0295) | -0.3749 (1) |
| − event head | 5 | 4.108 ± 0.054 | -0.043 | 8.062 ± 0.083 | 0.666 ± 0.004 | -0.0429\* (0.0365) | -2.5093\* (0.000165) |

DM: ablated minus headline per-window loss, averaged over seeds; **positive Δ means removing the component hurts**. `*` = p<0.05 after Holm over the six variants within this dataset and loss.
