# Every final number for the manuscript

Assembled by [analysis/numbers_for_paper.py](analysis/numbers_for_paper.py) from the generated artefacts. LaTeX blocks are byte-for-byte copies of `paper/tables/*.tex`; markdown tables are copied verbatim from the analysis reports; the combined accuracy table in §1 is rebuilt from `analysis/results_clean.csv` at 6 decimals. Nothing is retyped, rounded further or paraphrased.

Headline model: `XAI-MeteoFormer`, ablation `no_revin`, seq_len 96, HuberLoss(delta=1), 5 seeds. Baseline normalization is the validation-selected variant (`analysis/selection.py`); the historical variant is in `main_historical_*.tex`.

---

## 1. Main accuracy tables

### Jena — every metric, mean ± sd over 5 seeds

| Model | Norm. | params | MAE | RMSE | MSE | SMAPE | R2 | seeds |
|---|---|---|---|---|---|---|---|---|
| **MeteoFormer** | off (ours, ablation no_revin) | 1,892,475 | 3.024855 ± 0.030996 | 5.266825 ± 0.046154 | 27.741153 ± 0.486908 | 21.840793 ± 0.399999 | 0.681981 ± 0.004700 | 5 |
| Crossformer | off — validation rule keeps the historical variant | 1,695,908 | 3.107253 ± 0.017586 | 5.166186 ± 0.021308 | 26.689840 ± 0.220116 | 22.559869 ± 0.293513 | 0.678442 ± 0.004452 | 5 |
| iTransformer | off — validation rule keeps the historical variant | 1,611,032 | 3.294880 ± 0.011837 | 5.561181 ± 0.018098 | 30.926999 ± 0.201222 | 23.207196 ± 0.132086 | 0.626895 ± 0.002518 | 5 |
| TimesNet | off — validation rule keeps the historical variant | 1,186,507 | 3.310336 ± 0.016391 | 5.464023 ± 0.018482 | 29.855820 ± 0.201775 | 22.817160 ± 0.135496 | 0.646687 ± 0.003968 | 5 |
| LSTM | **off** — switched by the validation rule | 834,656 | 3.316641 ± 0.045025 | 5.410131 ± 0.056638 | 29.272084 ± 0.614487 | 23.321932 ± 0.193256 | 0.658358 ± 0.004286 | 5 |
| PatchTST | off — validation rule keeps the historical variant | 1,657,880 | 3.347684 ± 0.021425 | 5.635057 ± 0.033629 | 31.754774 ± 0.378836 | 23.158579 ± 0.118297 | 0.635906 ± 0.002347 | 5 |
| TFT | off — validation rule keeps the historical variant | 3,640,676 | 3.384346 ± 0.066168 | 5.551462 ± 0.053387 | 30.821004 ± 0.591456 | 23.124991 ± 0.225067 | 0.640603 ± 0.007805 | 5 |
| DLinear | off — validation rule keeps the historical variant | 4,656 | 3.423029 ± 0.009015 | 5.758403 ± 0.010538 | 33.159299 ± 0.121364 | 23.030659 ± 0.007838 | 0.616602 ± 0.000856 | 5 |
| Transformer | off — validation rule keeps the historical variant | 2,670,099 | 3.449454 ± 0.067656 | 5.627319 ± 0.085004 | 31.672498 ± 0.954820 | 23.639467 ± 0.214422 | 0.635159 ± 0.006974 | 5 |
| Informer | off — validation rule keeps the historical variant | 2,867,475 | 3.451844 ± 0.068491 | 5.631168 ± 0.100708 | 31.718172 ± 1.140591 | 23.777026 ± 0.416147 | 0.637583 ± 0.008749 | 5 |
| Autoformer | **on** — switched by the validation rule | 2,677,305 | 3.903827 ± 0.046879 | 6.203321 ± 0.069574 | 38.485062 ± 0.861217 | 25.282403 ± 0.281133 | 0.549323 ± 0.006761 | 5 |

### Beijing (Aotizhongxin) — every metric, mean ± sd over 5 seeds

| Model | Norm. | params | MAE | RMSE | MSE | SMAPE | R2 | seeds |
|---|---|---|---|---|---|---|---|---|
| **MeteoFormer** | off (ours, ablation no_revin) | 1,892,219 | 4.150543 ± 0.038510 | 8.215882 ± 0.048692 | 67.502608 ± 0.798442 | 26.693307 ± 0.229056 | 0.658624 ± 0.002470 | 5 |
| Crossformer | off — validation rule keeps the historical variant | 1,694,628 | 4.246992 ± 0.094692 | 8.002205 ± 0.172030 | 64.058959 ± 2.748563 | 27.508176 ± 0.442520 | 0.657408 ± 0.009974 | 5 |
| iTransformer | off — validation rule keeps the historical variant | 1,611,032 | 4.251023 ± 0.017274 | 8.265865 ± 0.032600 | 68.325368 ± 0.538478 | 27.313134 ± 0.076294 | 0.623825 ± 0.001236 | 5 |
| LSTM | on — validation rule keeps the historical variant | 833,668 | 4.342733 ± 0.033950 | 8.103445 ± 0.027327 | 65.666425 ± 0.443695 | 27.538537 ± 0.146192 | 0.637946 ± 0.002491 | 5 |
| DLinear | off — validation rule keeps the historical variant | 4,656 | 4.362253 ± 0.007909 | 8.380522 ± 0.012731 | 70.233278 ± 0.213415 | 27.739556 ± 0.206204 | 0.621243 ± 0.001150 | 5 |
| PatchTST | off — validation rule keeps the historical variant | 1,657,880 | 4.370325 ± 0.046200 | 8.336324 ± 0.049914 | 69.496289 ± 0.832224 | 27.559223 ± 0.170250 | 0.627846 ± 0.002516 | 5 |
| TimesNet | off — validation rule keeps the historical variant | 1,186,378 | 4.425953 ± 0.034172 | 8.273295 ± 0.064431 | 68.450734 ± 1.062014 | 27.460229 ± 0.185836 | 0.630638 ± 0.003432 | 5 |
| TFT | off — validation rule keeps the historical variant | 3,540,960 | 4.675807 ± 0.177680 | 8.431025 ± 0.172340 | 71.105945 ± 2.893160 | 28.382045 ± 0.415220 | 0.618301 ± 0.012323 | 5 |
| Transformer | **on** — switched by the validation rule | 2,668,342 | 4.894638 ± 0.070559 | 8.922561 ± 0.117890 | 79.623206 ± 2.103323 | 29.535975 ± 0.311380 | 0.592065 ± 0.003827 | 5 |
| Informer | off — validation rule keeps the historical variant | 2,865,682 | 4.931999 ± 0.090508 | 8.965073 ± 0.143206 | 80.388939 ± 2.587767 | 30.063701 ± 0.154676 | 0.594418 ± 0.003324 | 5 |
| Autoformer | **on** — switched by the validation rule | 2,674,742 | 5.347168 ± 0.112324 | 9.416733 ± 0.185684 | 88.702451 ± 3.487004 | 30.034226 ± 0.509993 | 0.550731 ± 0.013720 | 5 |

Per-target and per-horizon columns of the same rows (`MAE_T`, `RMSE_T`, `MAPE_P`, `MAE_h1` …) are in `analysis/results_clean.csv`; the tables in section 3 below are generated from them.

**`paper/tables/main_jena.tex`** — as typeset, MAE/RMSE/R² only

```latex
\begin{table}[H]
\caption{Forecasting accuracy on jena, mean $\pm$ s.d. over 5 seeds. Best per column in bold. $^{\dagger}$: with a RevIN layer added to the baseline. Each baseline's normalization is the variant with the lower mean validation loss, the rule that selected no RevIN for our model.}
\label{tab:main_jena}
\begin{tabular}{lccc}
\toprule
Model & MAE & RMSE & $R^2$ \\
\midrule
\textbf{MeteoFormer} & \textbf{3.025 $\pm$ 0.031} & 5.267 $\pm$ 0.046 & \textbf{0.682 $\pm$ 0.005} \\
Crossformer & 3.107 $\pm$ 0.018 & \textbf{5.166 $\pm$ 0.021} & 0.678 $\pm$ 0.004 \\
iTransformer & 3.295 $\pm$ 0.012 & 5.561 $\pm$ 0.018 & 0.627 $\pm$ 0.003 \\
TimesNet & 3.310 $\pm$ 0.016 & 5.464 $\pm$ 0.018 & 0.647 $\pm$ 0.004 \\
LSTM & 3.317 $\pm$ 0.045 & 5.410 $\pm$ 0.057 & 0.658 $\pm$ 0.004 \\
PatchTST & 3.348 $\pm$ 0.021 & 5.635 $\pm$ 0.034 & 0.636 $\pm$ 0.002 \\
TFT & 3.384 $\pm$ 0.066 & 5.551 $\pm$ 0.053 & 0.641 $\pm$ 0.008 \\
DLinear & 3.423 $\pm$ 0.009 & 5.758 $\pm$ 0.011 & 0.617 $\pm$ 0.001 \\
Transformer & 3.449 $\pm$ 0.068 & 5.627 $\pm$ 0.085 & 0.635 $\pm$ 0.007 \\
Informer & 3.452 $\pm$ 0.068 & 5.631 $\pm$ 0.101 & 0.638 $\pm$ 0.009 \\
Autoformer$^{\dagger}$ & 3.904 $\pm$ 0.047 & 6.203 $\pm$ 0.070 & 0.549 $\pm$ 0.007 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/main_beijing_aotizhongxin.tex`** — as typeset

```latex
\begin{table}[H]
\caption{Forecasting accuracy on beijing_aotizhongxin, mean $\pm$ s.d. over 5 seeds. Best per column in bold. $^{\dagger}$: with a RevIN layer added to the baseline. Each baseline's normalization is the variant with the lower mean validation loss, the rule that selected no RevIN for our model.}
\label{tab:main_beijing_aotizhongxin}
\begin{tabular}{lccc}
\toprule
Model & MAE & RMSE & $R^2$ \\
\midrule
\textbf{MeteoFormer} & \textbf{4.151 $\pm$ 0.039} & 8.216 $\pm$ 0.049 & \textbf{0.659 $\pm$ 0.002} \\
Crossformer & 4.247 $\pm$ 0.095 & \textbf{8.002 $\pm$ 0.172} & 0.657 $\pm$ 0.010 \\
iTransformer & 4.251 $\pm$ 0.017 & 8.266 $\pm$ 0.033 & 0.624 $\pm$ 0.001 \\
LSTM$^{\dagger}$ & 4.343 $\pm$ 0.034 & 8.103 $\pm$ 0.027 & 0.638 $\pm$ 0.002 \\
DLinear & 4.362 $\pm$ 0.008 & 8.381 $\pm$ 0.013 & 0.621 $\pm$ 0.001 \\
PatchTST & 4.370 $\pm$ 0.046 & 8.336 $\pm$ 0.050 & 0.628 $\pm$ 0.003 \\
TimesNet & 4.426 $\pm$ 0.034 & 8.273 $\pm$ 0.064 & 0.631 $\pm$ 0.003 \\
TFT & 4.676 $\pm$ 0.178 & 8.431 $\pm$ 0.172 & 0.618 $\pm$ 0.012 \\
Transformer$^{\dagger}$ & 4.895 $\pm$ 0.071 & 8.923 $\pm$ 0.118 & 0.592 $\pm$ 0.004 \\
Informer & 4.932 $\pm$ 0.091 & 8.965 $\pm$ 0.143 & 0.594 $\pm$ 0.003 \\
Autoformer$^{\dagger}$ & 5.347 $\pm$ 0.112 & 9.417 $\pm$ 0.186 & 0.551 $\pm$ 0.014 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/main_historical_jena.tex`** — appendix: historical normalization

```latex
\begin{table}[H]
\caption{Forecasting accuracy on jena, mean $\pm$ s.d. over 5 seeds. Best per column in bold. $^{\dagger}$: with a RevIN layer added to the baseline. Appendix: every baseline in its historical configuration (as its source defines it; LSTM with RevIN).}
\label{tab:main_historical_jena}
\begin{tabular}{lccc}
\toprule
Model & MAE & RMSE & $R^2$ \\
\midrule
\textbf{MeteoFormer} & \textbf{3.025 $\pm$ 0.031} & 5.267 $\pm$ 0.046 & \textbf{0.682 $\pm$ 0.005} \\
Crossformer & 3.107 $\pm$ 0.018 & \textbf{5.166 $\pm$ 0.021} & 0.678 $\pm$ 0.004 \\
iTransformer & 3.295 $\pm$ 0.012 & 5.561 $\pm$ 0.018 & 0.627 $\pm$ 0.003 \\
TimesNet & 3.310 $\pm$ 0.016 & 5.464 $\pm$ 0.018 & 0.647 $\pm$ 0.004 \\
PatchTST & 3.348 $\pm$ 0.021 & 5.635 $\pm$ 0.034 & 0.636 $\pm$ 0.002 \\
LSTM$^{\dagger}$ & 3.354 $\pm$ 0.022 & 5.560 $\pm$ 0.017 & 0.639 $\pm$ 0.002 \\
TFT & 3.384 $\pm$ 0.066 & 5.551 $\pm$ 0.053 & 0.641 $\pm$ 0.008 \\
DLinear & 3.423 $\pm$ 0.009 & 5.758 $\pm$ 0.011 & 0.617 $\pm$ 0.001 \\
Transformer & 3.449 $\pm$ 0.068 & 5.627 $\pm$ 0.085 & 0.635 $\pm$ 0.007 \\
Informer & 3.452 $\pm$ 0.068 & 5.631 $\pm$ 0.101 & 0.638 $\pm$ 0.009 \\
Autoformer & 3.989 $\pm$ 0.113 & 6.269 $\pm$ 0.178 & 0.548 $\pm$ 0.013 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/main_historical_beijing_aotizhongxin.tex`** — appendix: historical normalization

```latex
\begin{table}[H]
\caption{Forecasting accuracy on beijing_aotizhongxin, mean $\pm$ s.d. over 5 seeds. Best per column in bold. $^{\dagger}$: with a RevIN layer added to the baseline. Appendix: every baseline in its historical configuration (as its source defines it; LSTM with RevIN).}
\label{tab:main_historical_beijing_aotizhongxin}
\begin{tabular}{lccc}
\toprule
Model & MAE & RMSE & $R^2$ \\
\midrule
\textbf{MeteoFormer} & \textbf{4.151 $\pm$ 0.039} & 8.216 $\pm$ 0.049 & \textbf{0.659 $\pm$ 0.002} \\
Crossformer & 4.247 $\pm$ 0.095 & \textbf{8.002 $\pm$ 0.172} & 0.657 $\pm$ 0.010 \\
iTransformer & 4.251 $\pm$ 0.017 & 8.266 $\pm$ 0.033 & 0.624 $\pm$ 0.001 \\
LSTM$^{\dagger}$ & 4.343 $\pm$ 0.034 & 8.103 $\pm$ 0.027 & 0.638 $\pm$ 0.002 \\
DLinear & 4.362 $\pm$ 0.008 & 8.381 $\pm$ 0.013 & 0.621 $\pm$ 0.001 \\
PatchTST & 4.370 $\pm$ 0.046 & 8.336 $\pm$ 0.050 & 0.628 $\pm$ 0.003 \\
TimesNet & 4.426 $\pm$ 0.034 & 8.273 $\pm$ 0.064 & 0.631 $\pm$ 0.003 \\
TFT & 4.676 $\pm$ 0.178 & 8.431 $\pm$ 0.172 & 0.618 $\pm$ 0.012 \\
Informer & 4.932 $\pm$ 0.091 & 8.965 $\pm$ 0.143 & 0.594 $\pm$ 0.003 \\
Transformer & 4.974 $\pm$ 0.047 & 8.956 $\pm$ 0.170 & 0.597 $\pm$ 0.007 \\
Autoformer & 5.461 $\pm$ 0.035 & 9.551 $\pm$ 0.116 & 0.539 $\pm$ 0.009 \\
\bottomrule
\end{tabular}
\end{table}
```

## 2. Significance

**`paper/tables/significance_jena.tex`**

```latex
\begin{table}[H]
\caption{Diebold--Mariano test of MeteoFormer against each baseline on jena, on the per-window loss differential with a Newey--West HAC variance (automatic lag) and the Harvey--Leybourne--Newbold correction. $L_1$ = absolute loss, $L_2$ = squared loss; the per-window loss is averaged over the 5 seeds. Negative $\Delta$ favours MeteoFormer. $^{*}$: $p<0.05$ after Holm correction within each column. Baselines: each in the normalization variant with the lower mean validation loss (analysis/selection.py).}
\label{tab:sig_jena}
\begin{tabular}{lcccc}
\toprule
Baseline & $\Delta L_1$ & $p_{\mathrm{Holm}}$ & $\Delta L_2$ & $p_{\mathrm{Holm}}$ \\
\midrule
Autoformer & -0.879$^{*}$ & $<$0.001 & -10.744$^{*}$ & $<$0.001 \\
Crossformer & -0.082$^{*}$ & $<$0.001 & +1.051$^{*}$ & $<$0.001 \\
DLinear & -0.398$^{*}$ & $<$0.001 & -5.418$^{*}$ & $<$0.001 \\
Informer & -0.427$^{*}$ & $<$0.001 & -3.977$^{*}$ & $<$0.001 \\
LSTM & -0.292$^{*}$ & $<$0.001 & -1.531$^{*}$ & $<$0.001 \\
PatchTST & -0.323$^{*}$ & $<$0.001 & -4.014$^{*}$ & $<$0.001 \\
TFT & -0.359$^{*}$ & $<$0.001 & -3.080$^{*}$ & $<$0.001 \\
TimesNet & -0.285$^{*}$ & $<$0.001 & -2.115$^{*}$ & $<$0.001 \\
Transformer & -0.425$^{*}$ & $<$0.001 & -3.931$^{*}$ & $<$0.001 \\
iTransformer & -0.270$^{*}$ & $<$0.001 & -3.186$^{*}$ & $<$0.001 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/significance_beijing_aotizhongxin.tex`**

```latex
\begin{table}[H]
\caption{Diebold--Mariano test of MeteoFormer against each baseline on beijing_aotizhongxin, on the per-window loss differential with a Newey--West HAC variance (automatic lag) and the Harvey--Leybourne--Newbold correction. $L_1$ = absolute loss, $L_2$ = squared loss; the per-window loss is averaged over the 5 seeds. Negative $\Delta$ favours MeteoFormer. $^{*}$: $p<0.05$ after Holm correction within each column. Baselines: each in the normalization variant with the lower mean validation loss (analysis/selection.py).}
\label{tab:sig_beijing_aotizhongxin}
\begin{tabular}{lcccc}
\toprule
Baseline & $\Delta L_1$ & $p_{\mathrm{Holm}}$ & $\Delta L_2$ & $p_{\mathrm{Holm}}$ \\
\midrule
Autoformer & -1.197$^{*}$ & $<$0.001 & -21.200$^{*}$ & $<$0.001 \\
Crossformer & -0.096$^{*}$ & $<$0.001 & +3.444$^{*}$ & 0.003 \\
DLinear & -0.212$^{*}$ & $<$0.001 & -2.731 & 0.442 \\
Informer & -0.781$^{*}$ & $<$0.001 & -12.886$^{*}$ & $<$0.001 \\
LSTM & -0.192$^{*}$ & $<$0.001 & +1.836 & 0.631 \\
PatchTST & -0.220$^{*}$ & $<$0.001 & -1.994 & 0.631 \\
TFT & -0.525$^{*}$ & $<$0.001 & -3.603$^{*}$ & 0.014 \\
TimesNet & -0.275$^{*}$ & $<$0.001 & -0.948 & 1.000 \\
Transformer & -0.744$^{*}$ & $<$0.001 & -12.121$^{*}$ & $<$0.001 \\
iTransformer & -0.100$^{*}$ & 0.016 & -0.823 & 1.000 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/significance_channels_jena.tex`**

```latex
\begin{table}[H]
\caption{Diebold--Mariano differential per target channel on jena. Negative favours MeteoFormer. $^{*}$: $p<0.05$ after Holm correction within each column. The sign flips between temperature and relative humidity.}
\label{tab:sig_ch_jena}
\begin{tabular}{lcccccccc}
\toprule
& \multicolumn{4}{c}{$\Delta L_1$} & \multicolumn{4}{c}{$\Delta L_2$} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Baseline & T & RH & P & WS & T & RH & P & WS \\
\midrule
Autoformer & -0.778$^{*}$ & -1.205$^{*}$ & -1.433$^{*}$ & -0.101$^{*}$ & -5.646$^{*}$ & -22.944$^{*}$ & -14.055$^{*}$ & -0.331$^{*}$ \\
Crossformer & -0.577$^{*}$ & +0.225$^{*}$ & +0.007 & +0.015$^{*}$ & -3.525$^{*}$ & +7.517$^{*}$ & +0.195 & +0.018 \\
DLinear & -0.164$^{*}$ & -0.557$^{*}$ & -0.794$^{*}$ & -0.078$^{*}$ & -1.293$^{*}$ & -13.580$^{*}$ & -6.595$^{*}$ & -0.205$^{*}$ \\
Informer & -0.987$^{*}$ & -0.264$^{*}$ & -0.450$^{*}$ & -0.007$^{*}$ & -6.844$^{*}$ & -5.960$^{*}$ & -3.092$^{*}$ & -0.012 \\
LSTM & -0.783$^{*}$ & -0.139$^{*}$ & -0.236$^{*}$ & -0.009$^{*}$ & -4.932$^{*}$ & +0.244 & -1.437$^{*}$ & +0.002 \\
PatchTST & -0.405$^{*}$ & -0.601$^{*}$ & -0.228$^{*}$ & -0.057$^{*}$ & -2.832$^{*}$ & -11.415$^{*}$ & -1.646$^{*}$ & -0.161$^{*}$ \\
TFT & -0.548$^{*}$ & -0.396$^{*}$ & -0.466$^{*}$ & -0.027$^{*}$ & -3.703$^{*}$ & -5.086$^{*}$ & -3.444$^{*}$ & -0.086$^{*}$ \\
TimesNet & -0.471$^{*}$ & -0.270$^{*}$ & -0.373$^{*}$ & -0.028$^{*}$ & -3.060$^{*}$ & -2.527 & -2.775$^{*}$ & -0.097$^{*}$ \\
Transformer & -0.845$^{*}$ & -0.322$^{*}$ & -0.514$^{*}$ & -0.018$^{*}$ & -5.840$^{*}$ & -6.205$^{*}$ & -3.633$^{*}$ & -0.047$^{*}$ \\
iTransformer & -0.378$^{*}$ & -0.342$^{*}$ & -0.291$^{*}$ & -0.069$^{*}$ & -2.948$^{*}$ & -7.288$^{*}$ & -2.257$^{*}$ & -0.250$^{*}$ \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/significance_channels_beijing_aotizhongxin.tex`**

```latex
\begin{table}[H]
\caption{Diebold--Mariano differential per target channel on beijing_aotizhongxin. Negative favours MeteoFormer. $^{*}$: $p<0.05$ after Holm correction within each column. The sign flips between temperature and relative humidity.}
\label{tab:sig_ch_beijing_aotizhongxin}
\begin{tabular}{lcccccccc}
\toprule
& \multicolumn{4}{c}{$\Delta L_1$} & \multicolumn{4}{c}{$\Delta L_2$} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Baseline & T & RH & P & WS & T & RH & P & WS \\
\midrule
Autoformer & -1.120$^{*}$ & -1.939$^{*}$ & -1.683$^{*}$ & -0.044$^{*}$ & -9.228$^{*}$ & -58.658$^{*}$ & -16.772$^{*}$ & -0.141$^{*}$ \\
Crossformer & -1.024$^{*}$ & +0.580$^{*}$ & +0.041 & +0.016$^{*}$ & -7.853$^{*}$ & +21.682$^{*}$ & -0.064 & +0.010 \\
DLinear & -0.008 & -0.216 & -0.568$^{*}$ & -0.055$^{*}$ & -0.084 & -5.454 & -5.277$^{*}$ & -0.107$^{*}$ \\
Informer & -1.731$^{*}$ & -0.750$^{*}$ & -0.617$^{*}$ & -0.028$^{*}$ & -15.515$^{*}$ & -31.404$^{*}$ & -4.558$^{*}$ & -0.068$^{*}$ \\
LSTM & -0.557$^{*}$ & +0.141 & -0.328$^{*}$ & -0.025$^{*}$ & -4.335$^{*}$ & +14.681$^{*}$ & -2.939$^{*}$ & -0.062$^{*}$ \\
PatchTST & -0.163$^{*}$ & -0.385$^{*}$ & -0.297$^{*}$ & -0.034$^{*}$ & -1.250$^{*}$ & -3.857 & -2.772$^{*}$ & -0.096$^{*}$ \\
TFT & -0.945$^{*}$ & -0.409$^{*}$ & -0.712$^{*}$ & -0.036$^{*}$ & -7.556$^{*}$ & -0.168 & -6.627$^{*}$ & -0.062$^{*}$ \\
TimesNet & -0.695$^{*}$ & -0.019 & -0.371$^{*}$ & -0.017$^{*}$ & -5.370$^{*}$ & +5.218 & -3.578$^{*}$ & -0.063$^{*}$ \\
Transformer & -1.123$^{*}$ & -1.194$^{*}$ & -0.611$^{*}$ & -0.049$^{*}$ & -9.287$^{*}$ & -33.457$^{*}$ & -5.627$^{*}$ & -0.111$^{*}$ \\
iTransformer & -0.017 & +0.123 & -0.460$^{*}$ & -0.049$^{*}$ & -0.281 & +1.654 & -4.549$^{*}$ & -0.115$^{*}$ \\
\bottomrule
\end{tabular}
\end{table}
```

Raw values, every baseline × loss × channel, including DM statistic, lag and unadjusted p: `analysis/significance_dm.csv`. Comparison against the historical normalization instead: `analysis/significance_historical*`.

## 3. Per-target and per-horizon

**`paper/tables/per_target_mae_jena.tex`**

```latex
\begin{table}[H]
\caption{Per-target MAE on jena, mean $\pm$ s.d. over 5 seeds. Best per column in bold.}
\label{tab:pt_mae_jena}
\begin{tabular}{lcccc}
\toprule
Model & T ($^\circ$C) & RH (\%) & P (mbar) & WS (m\,s$^{-1}$) \\
\midrule
\textbf{MeteoFormer} & \textbf{1.953 $\pm$ 0.033} & 7.152 $\pm$ 0.064 & 2.145 $\pm$ 0.046 & 0.849 $\pm$ 0.003 \\
Autoformer & 2.731 $\pm$ 0.080 & 8.357 $\pm$ 0.150 & 3.578 $\pm$ 0.120 & 0.950 $\pm$ 0.010 \\
Crossformer & 2.531 $\pm$ 0.103 & \textbf{6.927 $\pm$ 0.061} & \textbf{2.138 $\pm$ 0.063} & \textbf{0.833 $\pm$ 0.008} \\
DLinear & 2.117 $\pm$ 0.005 & 7.709 $\pm$ 0.017 & 2.939 $\pm$ 0.021 & 0.927 $\pm$ 0.001 \\
Informer & 2.940 $\pm$ 0.217 & 7.416 $\pm$ 0.171 & 2.595 $\pm$ 0.084 & 0.855 $\pm$ 0.014 \\
LSTM & 2.737 $\pm$ 0.100 & 7.292 $\pm$ 0.097 & 2.381 $\pm$ 0.039 & 0.857 $\pm$ 0.007 \\
PatchTST & 2.359 $\pm$ 0.037 & 7.753 $\pm$ 0.063 & 2.373 $\pm$ 0.029 & 0.906 $\pm$ 0.007 \\
TFT & 2.502 $\pm$ 0.153 & 7.548 $\pm$ 0.095 & 2.611 $\pm$ 0.219 & 0.876 $\pm$ 0.004 \\
TimesNet & 2.424 $\pm$ 0.060 & 7.422 $\pm$ 0.049 & 2.518 $\pm$ 0.032 & 0.876 $\pm$ 0.004 \\
Transformer & 2.798 $\pm$ 0.117 & 7.474 $\pm$ 0.130 & 2.659 $\pm$ 0.094 & 0.866 $\pm$ 0.004 \\
iTransformer & 2.332 $\pm$ 0.061 & 7.494 $\pm$ 0.036 & 2.436 $\pm$ 0.051 & 0.918 $\pm$ 0.003 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/per_target_mae_beijing_aotizhongxin.tex`**

```latex
\begin{table}[H]
\caption{Per-target MAE on beijing_aotizhongxin, mean $\pm$ s.d. over 5 seeds. Best per column in bold.}
\label{tab:pt_mae_beijing_aotizhongxin}
\begin{tabular}{lcccc}
\toprule
Model & T ($^\circ$C) & RH (\%) & P (mbar) & WS (m\,s$^{-1}$) \\
\midrule
\textbf{MeteoFormer} & \textbf{2.019 $\pm$ 0.048} & 11.668 $\pm$ 0.107 & 2.249 $\pm$ 0.024 & 0.667 $\pm$ 0.006 \\
Autoformer & 3.139 $\pm$ 0.101 & 13.607 $\pm$ 0.273 & 3.932 $\pm$ 0.103 & 0.711 $\pm$ 0.009 \\
Crossformer & 3.043 $\pm$ 0.213 & \textbf{11.088 $\pm$ 0.243} & \textbf{2.207 $\pm$ 0.051} & \textbf{0.650 $\pm$ 0.003} \\
DLinear & 2.026 $\pm$ 0.035 & 11.884 $\pm$ 0.003 & 2.817 $\pm$ 0.006 & 0.722 $\pm$ 0.002 \\
Informer & 3.749 $\pm$ 0.281 & 12.418 $\pm$ 0.280 & 2.866 $\pm$ 0.222 & 0.694 $\pm$ 0.011 \\
LSTM & 2.576 $\pm$ 0.050 & 11.527 $\pm$ 0.090 & 2.576 $\pm$ 0.079 & 0.692 $\pm$ 0.006 \\
PatchTST & 2.182 $\pm$ 0.129 & 12.053 $\pm$ 0.123 & 2.546 $\pm$ 0.050 & 0.701 $\pm$ 0.003 \\
TFT & 2.964 $\pm$ 0.162 & 12.077 $\pm$ 0.337 & 2.961 $\pm$ 0.375 & 0.702 $\pm$ 0.006 \\
TimesNet & 2.714 $\pm$ 0.086 & 11.687 $\pm$ 0.125 & 2.620 $\pm$ 0.075 & 0.684 $\pm$ 0.004 \\
Transformer & 3.142 $\pm$ 0.211 & 12.862 $\pm$ 0.225 & 2.860 $\pm$ 0.040 & 0.716 $\pm$ 0.008 \\
iTransformer & 2.035 $\pm$ 0.035 & 11.545 $\pm$ 0.053 & 2.708 $\pm$ 0.035 & 0.716 $\pm$ 0.004 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/per_target_rmse_jena.tex`**

```latex
\begin{table}[H]
\caption{Per-target RMSE on jena, mean $\pm$ s.d. over 5 seeds. Best per column in bold.}
\label{tab:pt_rmse_jena}
\begin{tabular}{lcccc}
\toprule
Model & T ($^\circ$C) & RH (\%) & P (mbar) & WS (m\,s$^{-1}$) \\
\midrule
\textbf{MeteoFormer} & \textbf{2.547 $\pm$ 0.025} & 9.651 $\pm$ 0.092 & 3.138 $\pm$ 0.047 & 1.215 $\pm$ 0.009 \\
Autoformer & 3.483 $\pm$ 0.086 & 10.774 $\pm$ 0.136 & 4.887 $\pm$ 0.150 & 1.344 $\pm$ 0.009 \\
Crossformer & 3.162 $\pm$ 0.132 & \textbf{9.254 $\pm$ 0.036} & \textbf{3.107 $\pm$ 0.053} & \textbf{1.207 $\pm$ 0.007} \\
DLinear & 2.790 $\pm$ 0.008 & 10.331 $\pm$ 0.017 & 4.055 $\pm$ 0.024 & 1.296 $\pm$ 0.002 \\
Informer & 3.646 $\pm$ 0.225 & 9.954 $\pm$ 0.210 & 3.596 $\pm$ 0.093 & 1.220 $\pm$ 0.006 \\
LSTM & 3.378 $\pm$ 0.102 & 9.638 $\pm$ 0.111 & 3.359 $\pm$ 0.041 & 1.214 $\pm$ 0.002 \\
PatchTST & 3.053 $\pm$ 0.048 & 10.226 $\pm$ 0.059 & 3.390 $\pm$ 0.006 & 1.279 $\pm$ 0.004 \\
TFT & 3.189 $\pm$ 0.171 & 9.911 $\pm$ 0.088 & 3.640 $\pm$ 0.241 & 1.250 $\pm$ 0.009 \\
TimesNet & 3.090 $\pm$ 0.065 & 9.781 $\pm$ 0.044 & 3.553 $\pm$ 0.043 & 1.254 $\pm$ 0.008 \\
Transformer & 3.510 $\pm$ 0.106 & 9.967 $\pm$ 0.154 & 3.671 $\pm$ 0.090 & 1.234 $\pm$ 0.016 \\
iTransformer & 3.071 $\pm$ 0.078 & 10.022 $\pm$ 0.048 & 3.479 $\pm$ 0.057 & 1.314 $\pm$ 0.007 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/per_target_rmse_beijing_aotizhongxin.tex`**

```latex
\begin{table}[H]
\caption{Per-target RMSE on beijing_aotizhongxin, mean $\pm$ s.d. over 5 seeds. Best per column in bold.}
\label{tab:pt_rmse_beijing_aotizhongxin}
\begin{tabular}{lcccc}
\toprule
Model & T ($^\circ$C) & RH (\%) & P (mbar) & WS (m\,s$^{-1}$) \\
\midrule
\textbf{MeteoFormer} & \textbf{2.624 $\pm$ 0.050} & 15.887 $\pm$ 0.099 & \textbf{3.132 $\pm$ 0.030} & 0.946 $\pm$ 0.009 \\
Autoformer & 4.013 $\pm$ 0.112 & 17.635 $\pm$ 0.345 & 5.154 $\pm$ 0.118 & 1.018 $\pm$ 0.011 \\
Crossformer & 3.834 $\pm$ 0.233 & \textbf{15.187 $\pm$ 0.312} & 3.141 $\pm$ 0.060 & \textbf{0.941 $\pm$ 0.011} \\
DLinear & 2.640 $\pm$ 0.031 & 16.058 $\pm$ 0.030 & 3.884 $\pm$ 0.005 & 1.001 $\pm$ 0.002 \\
Informer & 4.726 $\pm$ 0.285 & 16.844 $\pm$ 0.386 & 3.785 $\pm$ 0.211 & 0.982 $\pm$ 0.009 \\
LSTM & 3.350 $\pm$ 0.053 & 15.419 $\pm$ 0.076 & 3.569 $\pm$ 0.090 & 0.979 $\pm$ 0.003 \\
PatchTST & 2.850 $\pm$ 0.146 & 16.008 $\pm$ 0.095 & 3.546 $\pm$ 0.066 & 0.995 $\pm$ 0.005 \\
TFT & 3.798 $\pm$ 0.168 & 15.891 $\pm$ 0.253 & 4.033 $\pm$ 0.465 & 0.978 $\pm$ 0.004 \\
TimesNet & 3.500 $\pm$ 0.098 & 15.722 $\pm$ 0.151 & 3.658 $\pm$ 0.077 & 0.979 $\pm$ 0.005 \\
Transformer & 4.017 $\pm$ 0.225 & 16.906 $\pm$ 0.267 & 3.928 $\pm$ 0.052 & 1.003 $\pm$ 0.004 \\
iTransformer & 2.677 $\pm$ 0.045 & 15.835 $\pm$ 0.075 & 3.789 $\pm$ 0.034 & 1.005 $\pm$ 0.002 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/per_target_r2_jena.tex`**

```latex
\begin{table}[H]
\caption{Per-target R2 on jena, mean $\pm$ s.d. over 5 seeds. Best per column in bold.}
\label{tab:pt_r2_jena}
\begin{tabular}{lcccc}
\toprule
Model & T & RH & P & WS \\
\midrule
\textbf{MeteoFormer} & \textbf{0.901 $\pm$ 0.002} & 0.647 $\pm$ 0.007 & 0.856 $\pm$ 0.004 & 0.324 $\pm$ 0.010 \\
Autoformer & 0.815 $\pm$ 0.009 & 0.561 $\pm$ 0.011 & 0.649 $\pm$ 0.021 & 0.172 $\pm$ 0.012 \\
Crossformer & 0.847 $\pm$ 0.013 & \textbf{0.676 $\pm$ 0.003} & \textbf{0.858 $\pm$ 0.005} & \textbf{0.332 $\pm$ 0.008} \\
DLinear & 0.881 $\pm$ 0.001 & 0.596 $\pm$ 0.001 & 0.759 $\pm$ 0.003 & 0.230 $\pm$ 0.002 \\
Informer & 0.797 $\pm$ 0.026 & 0.625 $\pm$ 0.016 & 0.810 $\pm$ 0.010 & 0.318 $\pm$ 0.006 \\
LSTM & 0.826 $\pm$ 0.010 & 0.648 $\pm$ 0.008 & 0.834 $\pm$ 0.004 & 0.325 $\pm$ 0.002 \\
PatchTST & 0.858 $\pm$ 0.004 & 0.604 $\pm$ 0.005 & 0.831 $\pm$ 0.001 & 0.250 $\pm$ 0.004 \\
TFT & 0.845 $\pm$ 0.016 & 0.628 $\pm$ 0.007 & 0.805 $\pm$ 0.026 & 0.285 $\pm$ 0.010 \\
TimesNet & 0.854 $\pm$ 0.006 & 0.638 $\pm$ 0.003 & 0.815 $\pm$ 0.004 & 0.280 $\pm$ 0.009 \\
Transformer & 0.812 $\pm$ 0.011 & 0.624 $\pm$ 0.012 & 0.802 $\pm$ 0.010 & 0.302 $\pm$ 0.018 \\
iTransformer & 0.856 $\pm$ 0.007 & 0.620 $\pm$ 0.004 & 0.822 $\pm$ 0.006 & 0.209 $\pm$ 0.008 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/per_target_r2_beijing_aotizhongxin.tex`**

```latex
\begin{table}[H]
\caption{Per-target R2 on beijing_aotizhongxin, mean $\pm$ s.d. over 5 seeds. Best per column in bold.}
\label{tab:pt_r2_beijing_aotizhongxin}
\begin{tabular}{lcccc}
\toprule
Model & T & RH & P & WS \\
\midrule
\textbf{MeteoFormer} & \textbf{0.955 $\pm$ 0.002} & 0.553 $\pm$ 0.006 & \textbf{0.917 $\pm$ 0.002} & 0.210 $\pm$ 0.014 \\
Autoformer & 0.894 $\pm$ 0.006 & 0.450 $\pm$ 0.021 & 0.774 $\pm$ 0.010 & 0.085 $\pm$ 0.021 \\
Crossformer & 0.903 $\pm$ 0.012 & \textbf{0.592 $\pm$ 0.017} & 0.916 $\pm$ 0.003 & \textbf{0.219 $\pm$ 0.018} \\
DLinear & 0.954 $\pm$ 0.001 & 0.544 $\pm$ 0.002 & 0.872 $\pm$ 0.000 & 0.115 $\pm$ 0.004 \\
Informer & 0.852 $\pm$ 0.018 & 0.498 $\pm$ 0.023 & 0.878 $\pm$ 0.013 & 0.149 $\pm$ 0.016 \\
LSTM & 0.926 $\pm$ 0.002 & 0.579 $\pm$ 0.004 & 0.892 $\pm$ 0.006 & 0.155 $\pm$ 0.006 \\
PatchTST & 0.946 $\pm$ 0.006 & 0.546 $\pm$ 0.005 & 0.893 $\pm$ 0.004 & 0.125 $\pm$ 0.009 \\
TFT & 0.905 $\pm$ 0.009 & 0.553 $\pm$ 0.014 & 0.860 $\pm$ 0.032 & 0.155 $\pm$ 0.006 \\
TimesNet & 0.919 $\pm$ 0.005 & 0.563 $\pm$ 0.008 & 0.886 $\pm$ 0.005 & 0.154 $\pm$ 0.008 \\
Transformer & 0.893 $\pm$ 0.012 & 0.494 $\pm$ 0.016 & 0.869 $\pm$ 0.003 & 0.112 $\pm$ 0.008 \\
iTransformer & 0.953 $\pm$ 0.002 & 0.556 $\pm$ 0.004 & 0.878 $\pm$ 0.002 & 0.108 $\pm$ 0.003 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/per_horizon_mae_jena.tex`**

```latex
\begin{table}[H]
\caption{Per-horizon MAE on jena (all four targets averaged), mean $\pm$ s.d. over 5 seeds. Best per column in bold.}
\label{tab:ph_mae_jena}
\begin{tabular}{lcccc}
\toprule
Model & $h$=1 & $h$=6 & $h$=12 & $h$=24 \\
\midrule
\textbf{MeteoFormer} & 1.208 $\pm$ 0.095 & \textbf{2.530 $\pm$ 0.045} & \textbf{3.200 $\pm$ 0.040} & \textbf{3.864 $\pm$ 0.043} \\
Autoformer & 2.865 $\pm$ 0.135 & 3.480 $\pm$ 0.063 & 3.929 $\pm$ 0.056 & 4.666 $\pm$ 0.026 \\
Crossformer & 1.492 $\pm$ 0.076 & 2.639 $\pm$ 0.039 & 3.226 $\pm$ 0.026 & 3.918 $\pm$ 0.018 \\
DLinear & \textbf{1.087 $\pm$ 0.057} & 2.982 $\pm$ 0.016 & 3.750 $\pm$ 0.006 & 4.199 $\pm$ 0.008 \\
Informer & 2.035 $\pm$ 0.068 & 3.029 $\pm$ 0.073 & 3.601 $\pm$ 0.075 & 4.267 $\pm$ 0.117 \\
LSTM & 1.882 $\pm$ 0.076 & 2.871 $\pm$ 0.047 & 3.445 $\pm$ 0.049 & 4.042 $\pm$ 0.033 \\
PatchTST & 1.284 $\pm$ 0.094 & 2.848 $\pm$ 0.028 & 3.590 $\pm$ 0.043 & 4.160 $\pm$ 0.022 \\
TFT & 1.877 $\pm$ 0.127 & 2.870 $\pm$ 0.079 & 3.510 $\pm$ 0.064 & 4.211 $\pm$ 0.102 \\
TimesNet & 1.834 $\pm$ 0.053 & 2.812 $\pm$ 0.016 & 3.439 $\pm$ 0.014 & 4.118 $\pm$ 0.036 \\
Transformer & 2.218 $\pm$ 0.148 & 3.021 $\pm$ 0.083 & 3.572 $\pm$ 0.067 & 4.148 $\pm$ 0.073 \\
iTransformer & 1.273 $\pm$ 0.031 & 2.756 $\pm$ 0.017 & 3.520 $\pm$ 0.018 & 4.179 $\pm$ 0.024 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/per_horizon_mae_beijing_aotizhongxin.tex`**

```latex
\begin{table}[H]
\caption{Per-horizon MAE on beijing_aotizhongxin (all four targets averaged), mean $\pm$ s.d. over 5 seeds. Best per column in bold.}
\label{tab:ph_mae_beijing_aotizhongxin}
\begin{tabular}{lcccc}
\toprule
Model & $h$=1 & $h$=6 & $h$=12 & $h$=24 \\
\midrule
\textbf{MeteoFormer} & 1.842 $\pm$ 0.133 & \textbf{3.448 $\pm$ 0.070} & \textbf{4.350 $\pm$ 0.054} & \textbf{5.207 $\pm$ 0.072} \\
Autoformer & 4.372 $\pm$ 0.173 & 4.843 $\pm$ 0.159 & 5.432 $\pm$ 0.109 & 6.181 $\pm$ 0.117 \\
Crossformer & 2.125 $\pm$ 0.125 & 3.615 $\pm$ 0.140 & 4.435 $\pm$ 0.082 & 5.281 $\pm$ 0.078 \\
DLinear & \textbf{1.545 $\pm$ 0.041} & 3.820 $\pm$ 0.019 & 4.746 $\pm$ 0.005 & 5.314 $\pm$ 0.015 \\
Informer & 3.387 $\pm$ 0.350 & 4.293 $\pm$ 0.113 & 5.032 $\pm$ 0.103 & 5.725 $\pm$ 0.056 \\
LSTM & 2.718 $\pm$ 0.091 & 3.728 $\pm$ 0.062 & 4.476 $\pm$ 0.062 & 5.301 $\pm$ 0.044 \\
PatchTST & 1.935 $\pm$ 0.223 & 3.809 $\pm$ 0.057 & 4.705 $\pm$ 0.032 & 5.290 $\pm$ 0.040 \\
TFT & 3.303 $\pm$ 0.274 & 4.080 $\pm$ 0.210 & 4.785 $\pm$ 0.175 & 5.575 $\pm$ 0.167 \\
TimesNet & 2.840 $\pm$ 0.135 & 3.801 $\pm$ 0.076 & 4.542 $\pm$ 0.009 & 5.388 $\pm$ 0.062 \\
Transformer & 3.023 $\pm$ 0.050 & 4.392 $\pm$ 0.194 & 5.126 $\pm$ 0.125 & 5.688 $\pm$ 0.084 \\
iTransformer & 1.833 $\pm$ 0.058 & 3.659 $\pm$ 0.037 & 4.543 $\pm$ 0.018 & 5.237 $\pm$ 0.020 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/per_horizon_rmse_jena.tex`**

```latex
\begin{table}[H]
\caption{Per-horizon RMSE on jena (all four targets averaged), mean $\pm$ s.d. over 5 seeds. Best per column in bold.}
\label{tab:ph_rmse_jena}
\begin{tabular}{lcccc}
\toprule
Model & $h$=1 & $h$=6 & $h$=12 & $h$=24 \\
\midrule
\textbf{MeteoFormer} & 2.091 $\pm$ 0.157 & 4.625 $\pm$ 0.107 & 5.502 $\pm$ 0.070 & 6.214 $\pm$ 0.069 \\
Autoformer & 4.469 $\pm$ 0.214 & 5.596 $\pm$ 0.072 & 6.267 $\pm$ 0.082 & 7.216 $\pm$ 0.084 \\
Crossformer & 2.297 $\pm$ 0.057 & \textbf{4.496 $\pm$ 0.036} & \textbf{5.352 $\pm$ 0.076} & \textbf{6.136 $\pm$ 0.024} \\
DLinear & \textbf{1.897 $\pm$ 0.085} & 5.119 $\pm$ 0.027 & 6.168 $\pm$ 0.014 & 6.596 $\pm$ 0.004 \\
Informer & 3.083 $\pm$ 0.097 & 5.035 $\pm$ 0.105 & 5.855 $\pm$ 0.128 & 6.719 $\pm$ 0.155 \\
LSTM & 2.972 $\pm$ 0.110 & 4.814 $\pm$ 0.062 & 5.579 $\pm$ 0.077 & 6.273 $\pm$ 0.035 \\
PatchTST & 2.135 $\pm$ 0.135 & 4.956 $\pm$ 0.027 & 5.959 $\pm$ 0.056 & 6.531 $\pm$ 0.038 \\
TFT & 2.978 $\pm$ 0.188 & 4.805 $\pm$ 0.072 & 5.708 $\pm$ 0.050 & 6.575 $\pm$ 0.130 \\
TimesNet & 3.001 $\pm$ 0.109 & 4.775 $\pm$ 0.033 & 5.631 $\pm$ 0.008 & 6.440 $\pm$ 0.047 \\
Transformer & 3.372 $\pm$ 0.240 & 5.035 $\pm$ 0.137 & 5.843 $\pm$ 0.113 & 6.475 $\pm$ 0.090 \\
iTransformer & 2.144 $\pm$ 0.037 & 4.829 $\pm$ 0.034 & 5.825 $\pm$ 0.015 & 6.570 $\pm$ 0.021 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/per_horizon_rmse_beijing_aotizhongxin.tex`**

```latex
\begin{table}[H]
\caption{Per-horizon RMSE on beijing_aotizhongxin (all four targets averaged), mean $\pm$ s.d. over 5 seeds. Best per column in bold.}
\label{tab:ph_rmse_beijing_aotizhongxin}
\begin{tabular}{lcccc}
\toprule
Model & $h$=1 & $h$=6 & $h$=12 & $h$=24 \\
\midrule
\textbf{MeteoFormer} & 3.487 $\pm$ 0.159 & 6.824 $\pm$ 0.097 & 8.478 $\pm$ 0.141 & 9.788 $\pm$ 0.139 \\
Autoformer & 7.377 $\pm$ 0.294 & 8.464 $\pm$ 0.271 & 9.543 $\pm$ 0.213 & 10.914 $\pm$ 0.195 \\
Crossformer & 3.563 $\pm$ 0.143 & \textbf{6.744 $\pm$ 0.193} & 8.340 $\pm$ 0.171 & \textbf{9.554 $\pm$ 0.256} \\
DLinear & \textbf{3.053 $\pm$ 0.032} & 7.285 $\pm$ 0.013 & 8.942 $\pm$ 0.016 & 9.686 $\pm$ 0.023 \\
Informer & 5.571 $\pm$ 0.592 & 7.778 $\pm$ 0.295 & 9.172 $\pm$ 0.190 & 10.253 $\pm$ 0.245 \\
LSTM & 4.795 $\pm$ 0.139 & 6.962 $\pm$ 0.066 & \textbf{8.333 $\pm$ 0.042} & 9.560 $\pm$ 0.083 \\
PatchTST & 3.579 $\pm$ 0.276 & 7.237 $\pm$ 0.072 & 8.827 $\pm$ 0.065 & 9.674 $\pm$ 0.041 \\
TFT & 5.684 $\pm$ 0.393 & 7.308 $\pm$ 0.220 & 8.617 $\pm$ 0.173 & 9.834 $\pm$ 0.170 \\
TimesNet & 5.061 $\pm$ 0.280 & 7.100 $\pm$ 0.104 & 8.462 $\pm$ 0.076 & 9.771 $\pm$ 0.044 \\
Transformer & 5.217 $\pm$ 0.096 & 8.040 $\pm$ 0.264 & 9.290 $\pm$ 0.235 & 10.087 $\pm$ 0.118 \\
iTransformer & 3.548 $\pm$ 0.119 & 7.106 $\pm$ 0.098 & 8.655 $\pm$ 0.011 & 9.758 $\pm$ 0.039 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/per_horizon_r2_jena.tex`**

```latex
\begin{table}[H]
\caption{Per-horizon R2 on jena (all four targets averaged), mean $\pm$ s.d. over 5 seeds. Best per column in bold.}
\label{tab:ph_r2_jena}
\begin{tabular}{lcccc}
\toprule
Model & $h$=1 & $h$=6 & $h$=12 & $h$=24 \\
\midrule
\textbf{MeteoFormer} & 0.927 $\pm$ 0.003 & \textbf{0.752 $\pm$ 0.006} & 0.660 $\pm$ 0.008 & \textbf{0.561 $\pm$ 0.007} \\
Autoformer & 0.748 $\pm$ 0.021 & 0.627 $\pm$ 0.007 & 0.545 $\pm$ 0.006 & 0.406 $\pm$ 0.010 \\
Crossformer & 0.908 $\pm$ 0.002 & 0.749 $\pm$ 0.007 & \textbf{0.663 $\pm$ 0.007} & 0.557 $\pm$ 0.006 \\
DLinear & \textbf{0.931 $\pm$ 0.002} & 0.695 $\pm$ 0.001 & 0.574 $\pm$ 0.001 & 0.491 $\pm$ 0.001 \\
Informer & 0.872 $\pm$ 0.004 & 0.710 $\pm$ 0.007 & 0.616 $\pm$ 0.010 & 0.501 $\pm$ 0.018 \\
LSTM & 0.874 $\pm$ 0.008 & 0.727 $\pm$ 0.004 & 0.641 $\pm$ 0.006 & 0.545 $\pm$ 0.005 \\
PatchTST & 0.920 $\pm$ 0.005 & 0.713 $\pm$ 0.002 & 0.606 $\pm$ 0.004 & 0.508 $\pm$ 0.004 \\
TFT & 0.877 $\pm$ 0.008 & 0.726 $\pm$ 0.007 & 0.625 $\pm$ 0.007 & 0.499 $\pm$ 0.018 \\
TimesNet & 0.875 $\pm$ 0.004 & 0.729 $\pm$ 0.002 & 0.631 $\pm$ 0.004 & 0.509 $\pm$ 0.007 \\
Transformer & 0.857 $\pm$ 0.013 & 0.704 $\pm$ 0.013 & 0.617 $\pm$ 0.007 & 0.516 $\pm$ 0.007 \\
iTransformer & 0.919 $\pm$ 0.001 & 0.716 $\pm$ 0.003 & 0.600 $\pm$ 0.004 & 0.480 $\pm$ 0.002 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/per_horizon_r2_beijing_aotizhongxin.tex`**

```latex
\begin{table}[H]
\caption{Per-horizon R2 on beijing_aotizhongxin (all four targets averaged), mean $\pm$ s.d. over 5 seeds. Best per column in bold.}
\label{tab:ph_r2_beijing_aotizhongxin}
\begin{tabular}{lcccc}
\toprule
Model & $h$=1 & $h$=6 & $h$=12 & $h$=24 \\
\midrule
\textbf{MeteoFormer} & 0.875 $\pm$ 0.005 & \textbf{0.722 $\pm$ 0.004} & \textbf{0.640 $\pm$ 0.005} & \textbf{0.562 $\pm$ 0.006} \\
Autoformer & 0.688 $\pm$ 0.017 & 0.609 $\pm$ 0.016 & 0.538 $\pm$ 0.015 & 0.444 $\pm$ 0.012 \\
Crossformer & 0.867 $\pm$ 0.005 & 0.719 $\pm$ 0.006 & 0.633 $\pm$ 0.013 & 0.559 $\pm$ 0.014 \\
DLinear & \textbf{0.883 $\pm$ 0.001} & 0.677 $\pm$ 0.001 & 0.581 $\pm$ 0.001 & 0.531 $\pm$ 0.002 \\
Informer & 0.785 $\pm$ 0.019 & 0.651 $\pm$ 0.006 & 0.579 $\pm$ 0.012 & 0.515 $\pm$ 0.012 \\
LSTM & 0.810 $\pm$ 0.007 & 0.697 $\pm$ 0.005 & 0.623 $\pm$ 0.004 & 0.543 $\pm$ 0.005 \\
PatchTST & 0.871 $\pm$ 0.006 & 0.688 $\pm$ 0.004 & 0.594 $\pm$ 0.005 & 0.533 $\pm$ 0.003 \\
TFT & 0.780 $\pm$ 0.019 & 0.680 $\pm$ 0.014 & 0.602 $\pm$ 0.012 & 0.525 $\pm$ 0.013 \\
TimesNet & 0.802 $\pm$ 0.012 & 0.694 $\pm$ 0.006 & 0.617 $\pm$ 0.005 & 0.530 $\pm$ 0.006 \\
Transformer & 0.803 $\pm$ 0.006 & 0.644 $\pm$ 0.013 & 0.567 $\pm$ 0.012 & 0.508 $\pm$ 0.008 \\
iTransformer & 0.870 $\pm$ 0.005 & 0.683 $\pm$ 0.004 & 0.592 $\pm$ 0.002 & 0.526 $\pm$ 0.002 \\
\bottomrule
\end{tabular}
\end{table}
```

**MAPE per target** — reported only for relative humidity and pressure; temperature in Celsius crosses zero, so a percentage error is undefined there and sMAPE (§1) is reported instead.

_Jena_

| Model | MAPE_RH | MAPE_P |
|---|---|---|
| **MeteoFormer** | 10.371116 ± 0.162372 | 0.216655 ± 0.004671 |
| Crossformer | 10.092515 ± 0.310643 | 0.215856 ± 0.006246 |
| iTransformer | 10.790029 ± 0.134018 | 0.246109 ± 0.005110 |
| TimesNet | 10.780586 ± 0.110234 | 0.254420 ± 0.003168 |
| LSTM | 10.551370 ± 0.169304 | 0.240453 ± 0.003951 |
| PatchTST | 11.426431 ± 0.132000 | 0.239739 ± 0.002878 |
| TFT | 11.015773 ± 0.415037 | 0.263797 ± 0.022110 |
| DLinear | 11.379673 ± 0.022783 | 0.296882 ± 0.002133 |
| Transformer | 10.757830 ± 0.207602 | 0.268495 ± 0.009402 |
| Informer | 10.695607 ± 0.252088 | 0.262182 ± 0.008536 |
| Autoformer | 12.138807 ± 0.355483 | 0.361316 ± 0.012103 |

_Beijing (Aotizhongxin)_

| Model | MAPE_RH | MAPE_P |
|---|---|---|
| **MeteoFormer** | 27.506128 ± 0.355917 | 0.221627 ± 0.002340 |
| Crossformer | 27.197920 ± 0.878783 | 0.217340 ± 0.005016 |
| iTransformer | 27.681915 ± 0.460486 | 0.266708 ± 0.003416 |
| LSTM | 27.028656 ± 0.623183 | 0.253855 ± 0.007789 |
| DLinear | 29.540262 ± 0.113459 | 0.277506 ± 0.000583 |
| PatchTST | 29.178083 ± 0.611611 | 0.250732 ± 0.004911 |
| TimesNet | 27.774319 ± 0.616131 | 0.257993 ± 0.007463 |
| TFT | 28.143770 ± 0.984637 | 0.291670 ± 0.036867 |
| Transformer | 30.819982 ± 0.565859 | 0.281716 ± 0.004016 |
| Informer | 30.110854 ± 0.630489 | 0.282521 ± 0.022037 |
| Autoformer | 32.568016 ± 0.794655 | 0.387657 ± 0.010072 |

## 4. Crossformer at full width, ablations, frost events

**`analysis/crossformer_fullwidth.md`**

## Accuracy by capacity

| Configuration | params | n seeds | MAE | RMSE | R² |
|---|---|---|---|---|---|
| Crossformer 128x128 | 1,695,908 | 5 | 3.107 ± 0.018 | 5.166 ± 0.021 | 0.678 ± 0.004 |
| Crossformer 256x512 | 7,981,860 | 5 | 3.184 ± 0.073 | 5.241 ± 0.063 | 0.675 ± 0.004 |
| Crossformer 256x1024 | 10,608,420 | 5 | 3.175 ± 0.042 | 5.237 ± 0.039 | 0.672 ± 0.005 |
| MeteoFormer (ours) | 1,892,475 | 5 | 3.025 ± 0.031 | 5.267 ± 0.046 | 0.682 ± 0.005 |

## Diebold–Mariano against the proposed model

| Configuration | Δ L1 | p | Δ L2 | p |
|---|---|---|---|---|
| Crossformer 128x128 | -0.0824* | 9.93e-12 | +1.0513* | 4.9e-07 |
| Crossformer 256x512 | -0.1595* | <1e-15 | +0.2659 | 0.273 |
| Crossformer 256x1024 | -0.1498* | <1e-15 | +0.3161 | 0.273 |

Negative Δ favours the proposed model. `*` = p<0.05 after Holm within each column. L1 = absolute loss, L2 = squared loss; the per-window loss is averaged over seeds, not the predictions.

## Result

- Crossformer 256x512: MAE 3.184 vs 3.107 at 128x128 (+0.077); vs ours 3.025 (-0.159).
- Crossformer 256x1024: MAE 3.175 vs 3.107 at 128x128 (+0.067); vs ours 3.025 (-0.150).

**`analysis/ablation_norevin.md`**

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

**`paper/tables/ablation_norevin.tex`** — from the headline configuration

```latex
\begin{table}[H]
\caption{Ablation from the headline configuration (no RevIN): each row removes one component, mean $\pm$ s.d. over seeds. $\Delta$MAE is relative to the headline model; positive means the component helps. Removing variable attention also removes the entropy term, which has no gradient once the attention is constant.}
\label{tab:ablation_norevin}
\begin{tabular}{llcccc}
\toprule
Dataset & Variant & $n$ & MAE & $\Delta$MAE & $R^2$ \\
\midrule
Jena & MeteoFormer (no RevIN, headline) & 5 & 3.025 $\pm$ 0.031 &  & 0.682 $\pm$ 0.005 \\
 & $-$ multi-scale & 5 & 3.041 $\pm$ 0.006 & +0.016 & 0.683 $\pm$ 0.002 \\
 & $-$ variable attention (and entropy term) & 5 & 3.100 $\pm$ 0.079 & +0.075 & 0.679 $\pm$ 0.006 \\
 & $-$ temporal attention & 5 & 3.035 $\pm$ 0.047 & +0.011 & 0.683 $\pm$ 0.005 \\
 & $-$ fusion gate & 5 & 3.092 $\pm$ 0.030 & +0.067 & 0.678 $\pm$ 0.003 \\
 & $-$ entropy term & 5 & 3.052 $\pm$ 0.052 & +0.027 & 0.683 $\pm$ 0.003 \\
 & $-$ event head & 5 & 3.052 $\pm$ 0.037 & +0.027 & 0.676 $\pm$ 0.005 \\
\midrule
Beijing & MeteoFormer (no RevIN, headline) & 5 & 4.151 $\pm$ 0.039 &  & 0.659 $\pm$ 0.002 \\
 & $-$ multi-scale & 5 & 4.161 $\pm$ 0.035 & +0.010 & 0.658 $\pm$ 0.008 \\
 & $-$ variable attention (and entropy term) & 5 & 4.157 $\pm$ 0.132 & +0.006 & 0.658 $\pm$ 0.007 \\
 & $-$ temporal attention & 5 & 4.163 $\pm$ 0.052 & +0.012 & 0.661 $\pm$ 0.005 \\
 & $-$ fusion gate & 5 & 4.194 $\pm$ 0.042 & +0.044 & 0.658 $\pm$ 0.003 \\
 & $-$ entropy term & 5 & 4.109 $\pm$ 0.088 & -0.042 & 0.662 $\pm$ 0.006 \\
 & $-$ event head & 5 & 4.108 $\pm$ 0.054 & -0.043 & 0.666 $\pm$ 0.004 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/ablation.tex`** — appendix: from `full`, Jena

```latex
\begin{table}[H]
\caption{Ablation on Jena, ordered by validation loss. The architecture was selected on validation, not on test. $n$ seeds is stated per row: the variants differ in how many were run.}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
Variant & $n$ seeds & Val.\ loss & MAE & RMSE & $R^2$ \\
\midrule
no revin & 5 & 0.1437 & 3.025 $\pm$ 0.031 & 5.267 $\pm$ 0.046 & 0.682 $\pm$ 0.005 \\
no multiscale & 5 & 0.1522 & 3.128 $\pm$ 0.026 & 5.459 $\pm$ 0.042 & 0.663 $\pm$ 0.003 \\
no cls & 5 & 0.1534 & 3.127 $\pm$ 0.024 & 5.470 $\pm$ 0.041 & 0.660 $\pm$ 0.004 \\
no entropy & 5 & 0.1556 & 3.178 $\pm$ 0.028 & 5.510 $\pm$ 0.047 & 0.657 $\pm$ 0.004 \\
no fusion & 5 & 0.1560 & 3.206 $\pm$ 0.028 & 5.578 $\pm$ 0.035 & 0.652 $\pm$ 0.003 \\
full & 5 & 0.1564 & 3.195 $\pm$ 0.019 & 5.536 $\pm$ 0.039 & 0.656 $\pm$ 0.004 \\
no var attn & 5 & 0.1568 & 3.183 $\pm$ 0.024 & 5.524 $\pm$ 0.053 & 0.656 $\pm$ 0.004 \\
no temp attn & 5 & 0.1569 & 3.199 $\pm$ 0.020 & 5.537 $\pm$ 0.025 & 0.654 $\pm$ 0.002 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/events_jena.tex`**

```latex
\begin{table}[H]
\caption{Frost-event detection on jena (T $\le$ 0\,$^\circ$C), mean $\pm$ s.d. over 5 seeds. Every model is scored from its predicted temperature at the physical freezing point, so no threshold is fitted. Positive rate 0.0722.}
\label{tab:events_jena}
\begin{tabular}{lccccc}
\toprule
Model & AUC & AP & F1 & Precision & Recall \\
\midrule
Crossformer & 0.980 $\pm$ 0.001 & 0.786 $\pm$ 0.009 & 0.717 $\pm$ 0.018 & 0.688 $\pm$ 0.077 & 0.764 $\pm$ 0.091 \\
LSTM & 0.976 $\pm$ 0.001 & 0.760 $\pm$ 0.007 & 0.705 $\pm$ 0.007 & 0.696 $\pm$ 0.021 & 0.716 $\pm$ 0.018 \\
\textbf{MeteoFormer} & 0.975 $\pm$ 0.002 & 0.757 $\pm$ 0.010 & 0.701 $\pm$ 0.009 & 0.667 $\pm$ 0.037 & 0.741 $\pm$ 0.028 \\
TimesNet & 0.975 $\pm$ 0.001 & 0.755 $\pm$ 0.010 & 0.702 $\pm$ 0.006 & 0.680 $\pm$ 0.029 & 0.726 $\pm$ 0.022 \\
Transformer & 0.976 $\pm$ 0.001 & 0.753 $\pm$ 0.017 & 0.706 $\pm$ 0.012 & 0.633 $\pm$ 0.018 & 0.798 $\pm$ 0.038 \\
Informer & 0.974 $\pm$ 0.001 & 0.740 $\pm$ 0.005 & 0.692 $\pm$ 0.019 & 0.606 $\pm$ 0.043 & 0.810 $\pm$ 0.032 \\
TFT & 0.971 $\pm$ 0.005 & 0.729 $\pm$ 0.036 & 0.675 $\pm$ 0.024 & 0.669 $\pm$ 0.051 & 0.688 $\pm$ 0.060 \\
PatchTST & 0.969 $\pm$ 0.001 & 0.710 $\pm$ 0.011 & 0.619 $\pm$ 0.020 & 0.730 $\pm$ 0.007 & 0.537 $\pm$ 0.028 \\
iTransformer & 0.969 $\pm$ 0.001 & 0.708 $\pm$ 0.005 & 0.640 $\pm$ 0.008 & 0.706 $\pm$ 0.012 & 0.586 $\pm$ 0.020 \\
DLinear & 0.965 $\pm$ 0.000 & 0.694 $\pm$ 0.001 & 0.553 $\pm$ 0.008 & 0.763 $\pm$ 0.010 & 0.434 $\pm$ 0.013 \\
Autoformer & 0.963 $\pm$ 0.001 & 0.649 $\pm$ 0.021 & 0.601 $\pm$ 0.018 & 0.612 $\pm$ 0.025 & 0.590 $\pm$ 0.023 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/events_beijing_aotizhongxin.tex`**

```latex
\begin{table}[H]
\caption{Frost-event detection on beijing_aotizhongxin (T $\le$ 0\,$^\circ$C), mean $\pm$ s.d. over 5 seeds. Every model is scored from its predicted temperature at the physical freezing point, so no threshold is fitted. Positive rate 0.1737.}
\label{tab:events_beijing_aotizhongxin}
\begin{tabular}{lccccc}
\toprule
Model & AUC & AP & F1 & Precision & Recall \\
\midrule
Crossformer & 0.977 $\pm$ 0.001 & 0.889 $\pm$ 0.006 & 0.817 $\pm$ 0.006 & 0.777 $\pm$ 0.030 & 0.865 $\pm$ 0.035 \\
\textbf{MeteoFormer} & 0.973 $\pm$ 0.002 & 0.872 $\pm$ 0.007 & 0.793 $\pm$ 0.018 & 0.776 $\pm$ 0.024 & 0.814 $\pm$ 0.060 \\
iTransformer & 0.973 $\pm$ 0.000 & 0.870 $\pm$ 0.002 & 0.762 $\pm$ 0.006 & 0.835 $\pm$ 0.003 & 0.700 $\pm$ 0.011 \\
Informer & 0.972 $\pm$ 0.001 & 0.869 $\pm$ 0.007 & 0.793 $\pm$ 0.002 & 0.729 $\pm$ 0.024 & 0.872 $\pm$ 0.038 \\
DLinear & 0.972 $\pm$ 0.001 & 0.867 $\pm$ 0.002 & 0.714 $\pm$ 0.023 & 0.869 $\pm$ 0.010 & 0.607 $\pm$ 0.037 \\
PatchTST & 0.970 $\pm$ 0.001 & 0.855 $\pm$ 0.009 & 0.748 $\pm$ 0.006 & 0.834 $\pm$ 0.006 & 0.679 $\pm$ 0.008 \\
TimesNet & 0.972 $\pm$ 0.001 & 0.855 $\pm$ 0.008 & 0.802 $\pm$ 0.006 & 0.775 $\pm$ 0.015 & 0.832 $\pm$ 0.020 \\
Transformer & 0.970 $\pm$ 0.001 & 0.852 $\pm$ 0.006 & 0.771 $\pm$ 0.015 & 0.801 $\pm$ 0.024 & 0.745 $\pm$ 0.050 \\
LSTM & 0.970 $\pm$ 0.002 & 0.847 $\pm$ 0.009 & 0.770 $\pm$ 0.010 & 0.803 $\pm$ 0.021 & 0.742 $\pm$ 0.037 \\
TFT & 0.968 $\pm$ 0.003 & 0.845 $\pm$ 0.015 & 0.784 $\pm$ 0.012 & 0.769 $\pm$ 0.022 & 0.801 $\pm$ 0.029 \\
Autoformer & 0.961 $\pm$ 0.003 & 0.815 $\pm$ 0.011 & 0.753 $\pm$ 0.008 & 0.766 $\pm$ 0.009 & 0.741 $\pm$ 0.009 \\
\bottomrule
\end{tabular}
\end{table}
```

## 5. Explanation fidelity (deterministic metric)

**`analysis/xai_fidelity_v2.md`**

## Jena

### Deterministic fidelity, mean ± sd over seeds

| Model | variant | n | SHAP-ranked | occlusion-ranked | ρ(SHAP, occl) |
|---|---|---|---|---|---|
| Crossformer | historical | 5 | 0.259 ± 0.028 | 0.265 ± 0.027 | 0.858 ± 0.101 |
| iTransformer | historical | 5 | 0.241 ± 0.006 | 0.256 ± 0.006 | 0.784 ± 0.088 |
| PatchTST | historical | 5 | 0.233 ± 0.004 | 0.243 ± 0.005 | 0.991 ± 0.003 |
| TimesNet | historical | 5 | 0.209 ± 0.019 | 0.256 ± 0.009 | 0.573 ± 0.168 |
| **MeteoFormer** | headline | 5 | 0.207 ± 0.007 | 0.211 ± 0.007 | 0.766 ± 0.130 |
| TFT | historical | 5 | 0.204 ± 0.038 | 0.232 ± 0.016 | 0.787 ± 0.098 |
| DLinear | historical | 5 | 0.186 ± 0.000 | 0.227 ± 0.001 | 0.747 ± 0.000 |
| LSTM | normoff | 5 | 0.146 ± 0.011 | 0.156 ± 0.009 | 0.841 ± 0.070 |
| Informer | historical | 5 | 0.135 ± 0.017 | 0.146 ± 0.018 | 0.774 ± 0.108 |
| Autoformer | normon | 5 | 0.125 ± 0.014 | 0.150 ± 0.013 | 0.598 ± 0.077 |
| Transformer | historical | 5 | 0.121 ± 0.017 | 0.135 ± 0.020 | 0.757 ± 0.106 |

**MeteoFormer, built-in attention ranking:** fidelity 0.012 ± 0.065; Spearman(attention, SHAP) 0.484 ± 0.335; Spearman(attention, occlusion) 0.371 ± 0.444.

### Paired tests against our model (SHAP-ranked, deterministic)

Paired by seed, paired t-test, Holm over the 10 baselines. Wilcoxon is not shown: with n = 5 its smallest two-sided p is 0.0625. Positive Δ = our model's SHAP ranking is more faithful.

| Baseline | Δ (ours − baseline) | paired t p | Holm | n |
|---|---|---|---|---|
| Crossformer | -0.052 | 0.0243 | 0.0729 | 5 |
| iTransformer | -0.034 | 0.000504 | 0.00453 **\*** | 5 |
| PatchTST | -0.026 | 0.00357 | 0.0143 **\*** | 5 |
| TimesNet | -0.002 | 0.779 | 1 | 5 |
| TFT | +0.003 | 0.868 | 1 | 5 |
| DLinear | +0.021 | 0.00282 | 0.0141 **\*** | 5 |
| LSTM | +0.061 | 0.00105 | 0.0072 **\*** | 5 |
| Informer | +0.072 | 0.00103 | 0.0072 **\*** | 5 |
| Autoformer | +0.082 | 0.000567 | 0.00454 **\*** | 5 |
| Transformer | +0.086 | 0.000422 | 0.00422 **\*** | 5 |

Our model's SHAP ranking is more faithful than 6 of 10 baselines on average; **7 of 10** differences survive Holm.


---

## Beijing (Aotizhongxin)

### Deterministic fidelity, mean ± sd over seeds

| Model | variant | n | SHAP-ranked | occlusion-ranked | ρ(SHAP, occl) |
|---|---|---|---|---|---|
| Crossformer | historical | 5 | 0.169 ± 0.016 | 0.185 ± 0.018 | 0.807 ± 0.082 |
| PatchTST | historical | 5 | 0.158 ± 0.011 | 0.176 ± 0.007 | 0.983 ± 0.010 |
| DLinear | historical | 5 | 0.144 ± 0.001 | 0.190 ± 0.002 | 0.760 ± 0.000 |
| iTransformer | historical | 5 | 0.142 ± 0.003 | 0.189 ± 0.003 | 0.524 ± 0.057 |
| **MeteoFormer** | headline | 5 | 0.130 ± 0.017 | 0.149 ± 0.012 | 0.669 ± 0.191 |
| TimesNet | historical | 5 | 0.109 ± 0.036 | 0.172 ± 0.007 | 0.655 ± 0.059 |
| TFT | historical | 5 | 0.104 ± 0.014 | 0.123 ± 0.016 | 0.744 ± 0.053 |
| LSTM | historical | 5 | 0.075 ± 0.013 | 0.167 ± 0.007 | 0.627 ± 0.103 |
| Autoformer | normon | 5 | 0.068 ± 0.007 | 0.077 ± 0.010 | 0.568 ± 0.139 |
| Informer | historical | 5 | 0.056 ± 0.014 | 0.072 ± 0.011 | 0.505 ± 0.121 |
| Transformer | normon | 5 | 0.042 ± 0.016 | 0.141 ± 0.019 | 0.321 ± 0.118 |

**MeteoFormer, built-in attention ranking:** fidelity 0.046 ± 0.092; Spearman(attention, SHAP) 0.266 ± 0.391; Spearman(attention, occlusion) 0.138 ± 0.403.

### Paired tests against our model (SHAP-ranked, deterministic)

Paired by seed, paired t-test, Holm over the 10 baselines. Wilcoxon is not shown: with n = 5 its smallest two-sided p is 0.0625. Positive Δ = our model's SHAP ranking is more faithful.

| Baseline | Δ (ours − baseline) | paired t p | Holm | n |
|---|---|---|---|---|
| Crossformer | -0.040 | 0.00311 | 0.0187 **\*** | 5 |
| PatchTST | -0.029 | 0.035 | 0.175 | 5 |
| DLinear | -0.014 | 0.135 | 0.404 | 5 |
| iTransformer | -0.012 | 0.159 | 0.404 | 5 |
| TimesNet | +0.021 | 0.168 | 0.404 | 5 |
| TFT | +0.026 | 0.0766 | 0.306 | 5 |
| LSTM | +0.055 | 0.000528 | 0.00422 **\*** | 5 |
| Autoformer | +0.061 | 0.00154 | 0.0107 **\*** | 5 |
| Informer | +0.074 | 0.000301 | 0.00301 **\*** | 5 |
| Transformer | +0.088 | 0.000341 | 0.00307 **\*** | 5 |

Our model's SHAP ranking is more faithful than 6 of 10 baselines on average; **5 of 10** differences survive Holm.


---

**`analysis/xai_fidelity_v2.md`**

## P0-4: does the entropy regulariser buy faithfulness?

Isolated by two variants that differ in `lambda_ent` alone, on the published configuration: `no_revin` (λ=0.01, the headline) vs `no_revin+no_entropy` (λ=0), both RevIN off, 5 seeds, both datasets, paired by seed. Deterministic metric; the time-axis rows come from [occlusion_time.md](analysis/occlusion_time.md), which was deterministic from the start.

> Correction. An earlier version compared `no_entropy` (RevIN on) with `no_revin` (RevIN off) and concluded that the regulariser doubles the attention–occlusion agreement. That comparison changed RevIN too and is withdrawn.

### Jena

| Metric (n = 5) | λ=0.01 (headline) | λ=0 | Δ | paired t p | λ>0 better |
|---|---|---|---|---|---|
| fidelity, attention-ranked | 0.012 ± 0.065 | 0.103 ± 0.055 | -0.091 | 0.099 | 1/5 |
| Spearman(attention, SHAP) | 0.484 ± 0.335 | 0.301 ± 0.233 | +0.183 | 0.203 | 4/5 |
| Spearman(attention, occlusion) | 0.371 ± 0.444 | 0.468 ± 0.253 | -0.097 | 0.436 | 2/5 |
| fidelity, SHAP-ranked | 0.207 ± 0.007 | 0.201 ± 0.012 | +0.006 | 0.235 | 4/5 |
| fidelity, occlusion-ranked | 0.211 ± 0.007 | 0.207 ± 0.013 | +0.005 | 0.377 | 3/5 |
| time occlusion ρ(attention) | 0.578 ± 0.215 | 0.471 ± 0.508 | +0.107 | 0.712 | 2/5 |

Recency control on the same windows: ρ = 0.582 ± 0.205.

### Beijing (Aotizhongxin)

| Metric (n = 5) | λ=0.01 (headline) | λ=0 | Δ | paired t p | λ>0 better |
|---|---|---|---|---|---|
| fidelity, attention-ranked | 0.046 ± 0.092 | 0.105 ± 0.016 | -0.060 | 0.161 | 1/5 |
| Spearman(attention, SHAP) | 0.266 ± 0.391 | 0.816 ± 0.031 | -0.550 | 0.029 | 0/5 |
| Spearman(attention, occlusion) | 0.138 ± 0.403 | 0.771 ± 0.041 | -0.634 | 0.027 | 0/5 |
| fidelity, SHAP-ranked | 0.130 ± 0.017 | 0.130 ± 0.013 | -0.000 | 0.963 | 3/5 |
| fidelity, occlusion-ranked | 0.149 ± 0.012 | 0.142 ± 0.010 | +0.007 | 0.146 | 4/5 |
| time occlusion ρ(attention) | 0.622 ± 0.394 | 0.567 ± 0.248 | +0.055 | 0.719 | 2/5 |

Recency control on the same windows: ρ = 0.407 ± 0.401.

**Multiple comparisons.** 12 paired tests. Smallest raw p = 0.027 (Beijing (Aotizhongxin), Spearman(attention, occlusion)), Holm-adjusted 0.33. No difference survives the correction.

**Reading.** On the attention rows λ=0.01 comes out ahead in **12 of 40** seed-level comparisons. A regulariser that made attention more faithful would win most of them. The SHAP- and occlusion-ranked rows describe the model rather than its attention and should barely move.

**Conclusion for Major #7.** The entropy regulariser does not improve faithfulness; no difference survives correction and the direction on the attention rows is against it. The claim that it "lifts the fidelity/stability numbers" (also in the docstring of [src/xm_models/xai_meteoformer.py](src/xm_models/xai_meteoformer.py)) is not supported. What it demonstrably does is sparsify the attention ([results_hygiene.md](analysis/results_hygiene.md)); if kept, describe it as a sparsity prior only.


---

**`analysis/xai_fidelity_v2.md`**

## What does move faithfulness: RevIN

`full` (RevIN on) vs the headline `no_revin` (RevIN off), 5 seeds each, paired by seed, deterministic metric.

### Jena

| Metric | RevIN on (full) | RevIN off (headline) | Δ | paired t p | on > off |
|---|---|---|---|---|---|
| fidelity, attention-ranked | 0.181 ± 0.079 | 0.012 ± 0.065 | +0.169 | 0.001 | 5/5 |
| Spearman(attention, SHAP) | 0.839 ± 0.060 | 0.484 ± 0.335 | +0.354 | 0.072 | 5/5 |
| Spearman(attention, occlusion) | 0.514 ± 0.129 | 0.371 ± 0.444 | +0.144 | 0.429 | 2/5 |
| fidelity, SHAP-ranked | 0.240 ± 0.013 | 0.207 ± 0.007 | +0.033 | 0.006 | 5/5 |
| fidelity, occlusion-ranked | 0.260 ± 0.014 | 0.211 ± 0.007 | +0.048 | 0.004 | 5/5 |

### Beijing (Aotizhongxin)

| Metric | RevIN on (full) | RevIN off (headline) | Δ | paired t p | on > off |
|---|---|---|---|---|---|
| fidelity, attention-ranked | 0.098 ± 0.035 | 0.046 ± 0.092 | +0.053 | 0.398 | 4/5 |
| Spearman(attention, SHAP) | 0.425 ± 0.216 | 0.266 ± 0.391 | +0.159 | 0.521 | 3/5 |
| Spearman(attention, occlusion) | 0.299 ± 0.141 | 0.138 ± 0.403 | +0.162 | 0.448 | 3/5 |
| fidelity, SHAP-ranked | 0.136 ± 0.022 | 0.130 ± 0.017 | +0.006 | 0.687 | 3/5 |
| fidelity, occlusion-ranked | 0.177 ± 0.013 | 0.149 ± 0.012 | +0.028 | 0.038 | 4/5 |


---

**`paper/tables/fidelity_jena.tex`**

```latex
\begin{table}[H]
\caption{Explanation fidelity on jena, physical channels only, mean $\pm$ s.d. over 5 seeds for every model. Deterministic metric: a perturbed channel is replaced by its own window mean and the reference ordering set is fixed and shared, so the spread is model variance only. Baselines in the normalization variant selected on validation. SHAP-ranked = fidelity of the model's GradientSHAP ranking; occlusion-ranked ranks by the scoring perturbation itself and serves as an upper reference.}
\label{tab:fid_jena}
\begin{tabular}{lccc}
\toprule
Model & SHAP-ranked & Occlusion-ranked & $\rho$(SHAP, occl.) \\
\midrule
Crossformer & 0.259 $\pm$ 0.028 & 0.265 $\pm$ 0.027 & 0.858 $\pm$ 0.101 \\
iTransformer & 0.241 $\pm$ 0.006 & 0.256 $\pm$ 0.006 & 0.784 $\pm$ 0.088 \\
PatchTST & 0.233 $\pm$ 0.004 & 0.243 $\pm$ 0.005 & 0.991 $\pm$ 0.003 \\
TimesNet & 0.209 $\pm$ 0.019 & 0.256 $\pm$ 0.009 & 0.573 $\pm$ 0.168 \\
\textbf{MeteoFormer} & 0.207 $\pm$ 0.007 & 0.211 $\pm$ 0.007 & 0.766 $\pm$ 0.130 \\
TFT & 0.204 $\pm$ 0.038 & 0.232 $\pm$ 0.016 & 0.787 $\pm$ 0.098 \\
DLinear & 0.186 $\pm$ 0.000 & 0.227 $\pm$ 0.001 & 0.747 $\pm$ 0.000 \\
LSTM & 0.146 $\pm$ 0.011 & 0.156 $\pm$ 0.009 & 0.841 $\pm$ 0.070 \\
Informer & 0.135 $\pm$ 0.017 & 0.146 $\pm$ 0.018 & 0.774 $\pm$ 0.108 \\
Autoformer & 0.125 $\pm$ 0.014 & 0.150 $\pm$ 0.013 & 0.598 $\pm$ 0.077 \\
Transformer & 0.121 $\pm$ 0.017 & 0.135 $\pm$ 0.020 & 0.757 $\pm$ 0.106 \\
\midrule
\textbf{MeteoFormer} (built-in attention) & \multicolumn{3}{c}{0.012 $\pm$ 0.065} \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/fidelity_beijing_aotizhongxin.tex`**

```latex
\begin{table}[H]
\caption{Explanation fidelity on beijing_aotizhongxin, physical channels only, mean $\pm$ s.d. over 5 seeds for every model. Deterministic metric: a perturbed channel is replaced by its own window mean and the reference ordering set is fixed and shared, so the spread is model variance only. Baselines in the normalization variant selected on validation. SHAP-ranked = fidelity of the model's GradientSHAP ranking; occlusion-ranked ranks by the scoring perturbation itself and serves as an upper reference.}
\label{tab:fid_beijing_aotizhongxin}
\begin{tabular}{lccc}
\toprule
Model & SHAP-ranked & Occlusion-ranked & $\rho$(SHAP, occl.) \\
\midrule
Crossformer & 0.169 $\pm$ 0.016 & 0.185 $\pm$ 0.018 & 0.807 $\pm$ 0.082 \\
PatchTST & 0.158 $\pm$ 0.011 & 0.176 $\pm$ 0.007 & 0.983 $\pm$ 0.010 \\
DLinear & 0.144 $\pm$ 0.001 & 0.190 $\pm$ 0.002 & 0.760 $\pm$ 0.000 \\
iTransformer & 0.142 $\pm$ 0.003 & 0.189 $\pm$ 0.003 & 0.524 $\pm$ 0.057 \\
\textbf{MeteoFormer} & 0.130 $\pm$ 0.017 & 0.149 $\pm$ 0.012 & 0.669 $\pm$ 0.191 \\
TimesNet & 0.109 $\pm$ 0.036 & 0.172 $\pm$ 0.007 & 0.655 $\pm$ 0.059 \\
TFT & 0.104 $\pm$ 0.014 & 0.123 $\pm$ 0.016 & 0.744 $\pm$ 0.053 \\
LSTM & 0.075 $\pm$ 0.013 & 0.167 $\pm$ 0.007 & 0.627 $\pm$ 0.103 \\
Autoformer & 0.068 $\pm$ 0.007 & 0.077 $\pm$ 0.010 & 0.568 $\pm$ 0.139 \\
Informer & 0.056 $\pm$ 0.014 & 0.072 $\pm$ 0.011 & 0.505 $\pm$ 0.121 \\
Transformer & 0.042 $\pm$ 0.016 & 0.141 $\pm$ 0.019 & 0.321 $\pm$ 0.118 \\
\midrule
\textbf{MeteoFormer} (built-in attention) & \multicolumn{3}{c}{0.046 $\pm$ 0.092} \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/fidelity_perm_appendix_jena.tex`** — appendix: permutation metric

```latex
\begin{table}[H]
\caption{Appendix. Explanation fidelity on jena with the permutation metric (random time permutation, single random reference ordering), 5 seeds. On one fixed checkpoint this estimator has s.d. $\approx$ 0.18 from its own sampling noise at this budget, so differences below $\approx$ 0.2 are not interpretable; the deterministic metric in the main text removes this noise.}
\label{tab:fid_perm_jena}
\begin{tabular}{lccc}
\toprule
Model & SHAP-ranked & Permutation-ranked & $\rho$(SHAP, perm.) \\
\midrule
Crossformer & 0.420 $\pm$ 0.194 & 0.413 $\pm$ 0.182 & 0.834 $\pm$ 0.080 \\
\textbf{MeteoFormer} & 0.401 $\pm$ 0.189 & 0.382 $\pm$ 0.178 & 0.779 $\pm$ 0.114 \\
iTransformer & 0.281 $\pm$ 0.127 & 0.299 $\pm$ 0.128 & 0.823 $\pm$ 0.063 \\
DLinear & 0.266 $\pm$ 0.106 & 0.301 $\pm$ 0.108 & 0.749 $\pm$ 0.003 \\
PatchTST & 0.265 $\pm$ 0.115 & 0.276 $\pm$ 0.117 & 0.991 $\pm$ 0.003 \\
TFT & 0.186 $\pm$ 0.110 & 0.221 $\pm$ 0.097 & 0.759 $\pm$ 0.054 \\
LSTM & 0.166 $\pm$ 0.075 & 0.182 $\pm$ 0.086 & 0.836 $\pm$ 0.093 \\
Informer & 0.161 $\pm$ 0.078 & 0.172 $\pm$ 0.084 & 0.820 $\pm$ 0.027 \\
TimesNet & 0.146 $\pm$ 0.093 & 0.175 $\pm$ 0.096 & 0.389 $\pm$ 0.256 \\
Transformer & 0.135 $\pm$ 0.068 & 0.150 $\pm$ 0.071 & 0.753 $\pm$ 0.122 \\
Autoformer & 0.090 $\pm$ 0.080 & 0.161 $\pm$ 0.065 & 0.410 $\pm$ 0.108 \\
\bottomrule
\end{tabular}
\end{table}
```

**`paper/tables/fidelity_perm_appendix_beijing_aotizhongxin.tex`** — appendix: permutation metric

```latex
\begin{table}[H]
\caption{Appendix. Explanation fidelity on beijing_aotizhongxin with the permutation metric (random time permutation, single random reference ordering), 5 seeds. On one fixed checkpoint this estimator has s.d. $\approx$ 0.18 from its own sampling noise at this budget, so differences below $\approx$ 0.2 are not interpretable; the deterministic metric in the main text removes this noise.}
\label{tab:fid_perm_beijing_aotizhongxin}
\begin{tabular}{lccc}
\toprule
Model & SHAP-ranked & Permutation-ranked & $\rho$(SHAP, perm.) \\
\midrule
iTransformer & 0.273 $\pm$ 0.084 & 0.318 $\pm$ 0.084 & 0.533 $\pm$ 0.040 \\
Crossformer & 0.269 $\pm$ 0.079 & 0.283 $\pm$ 0.079 & 0.854 $\pm$ 0.022 \\
DLinear & 0.257 $\pm$ 0.070 & 0.294 $\pm$ 0.071 & 0.760 $\pm$ 0.000 \\
\textbf{MeteoFormer} & 0.235 $\pm$ 0.095 & 0.251 $\pm$ 0.094 & 0.664 $\pm$ 0.136 \\
PatchTST & 0.224 $\pm$ 0.067 & 0.241 $\pm$ 0.067 & 0.986 $\pm$ 0.013 \\
TimesNet & 0.129 $\pm$ 0.074 & 0.147 $\pm$ 0.068 & 0.683 $\pm$ 0.084 \\
LSTM & 0.109 $\pm$ 0.052 & 0.137 $\pm$ 0.049 & 0.667 $\pm$ 0.099 \\
TFT & 0.101 $\pm$ 0.039 & 0.108 $\pm$ 0.044 & 0.685 $\pm$ 0.086 \\
Informer & 0.079 $\pm$ 0.045 & 0.098 $\pm$ 0.048 & 0.608 $\pm$ 0.180 \\
Transformer & 0.051 $\pm$ 0.039 & 0.108 $\pm$ 0.024 & 0.135 $\pm$ 0.172 \\
Autoformer & 0.035 $\pm$ 0.026 & 0.083 $\pm$ 0.042 & 0.160 $\pm$ 0.173 \\
\bottomrule
\end{tabular}
\end{table}
```

## 6. Robustness, transfer, window length, calibration

**`analysis/occlusion_time.md`** — all four configurations: headline, the clean entropy control, full, no_entropy

## Headline model (no_revin: RevIN off, entropy on) — Jena, 5 seeds

### Spearman agreement with the occlusion ranking

| | mean ρ | sd | per seed |
|---|---|---|---|
| temporal attention | +0.578 | 0.215 | +0.409, +0.673, +0.355, +0.564, +0.891 |
| attention rollout | +0.456 | 0.194 | +0.536, +0.473, +0.564, +0.591, +0.118 |
| **recency control** | +0.582 | 0.205 | +0.527, +0.527, +0.300, +0.718, +0.836 |

- paired attention − recency: **-0.0036** (t-test p = 0.952)
- paired rollout − recency: **-0.1255** (t-test p = 0.482)

### Where the model actually looks

| patch | hours | occlusion ΔMAE | temporal attention | rollout |
|---|---|---|---|---|
| p0 | 0-16 | **-0.0021** ± 0.0210 | 0.0905 ± 0.0004 | 0.0644 ± 0.0130 |
| p1 | 8-24 | **-0.0092** ± 0.0285 | 0.0906 ± 0.0002 | 0.0541 ± 0.0035 |
| p2 | 16-32 | **-0.0020** ± 0.0078 | 0.0905 ± 0.0001 | 0.0644 ± 0.0023 |
| p3 | 24-40 | **-0.0006** ± 0.0130 | 0.0904 ± 0.0002 | 0.0651 ± 0.0043 |
| p4 | 32-48 | **-0.0055** ± 0.0096 | 0.0907 ± 0.0001 | 0.0560 ± 0.0037 |
| p5 | 40-56 | **-0.0075** ± 0.0105 | 0.0907 ± 0.0002 | 0.0611 ± 0.0026 |
| p6 | 48-64 | **-0.0135** ± 0.0084 | 0.0907 ± 0.0001 | 0.0601 ± 0.0048 |
| p7 | 56-72 | **+0.0138** ± 0.0134 | 0.0910 ± 0.0001 | 0.0519 ± 0.0043 |
| p8 | 64-80 | **+0.0330** ± 0.0342 | 0.0911 ± 0.0003 | 0.0624 ± 0.0032 |
| p9 | 72-88 | **+0.1535** ± 0.1056 | 0.0915 ± 0.0002 | 0.0686 ± 0.0012 |
| p10 | 80-96 | **+2.0394** ± 0.1338 | 0.0924 ± 0.0006 | 0.3920 ± 0.0369 |

Attention spread: **0.0904 … 0.0924**, i.e. 2.1% of its own mean, against a uniform value of 1/11 = 0.0909. Occlusion importance spans -0.0135 … +2.0394.


---

## no_revin+no_entropy (RevIN off, entropy off) — clean control for the regulariser — Jena, 5 seeds

### Spearman agreement with the occlusion ranking

| | mean ρ | sd | per seed |
|---|---|---|---|
| temporal attention | +0.471 | 0.508 | +0.682, -0.436, +0.709, +0.745, +0.655 |
| attention rollout | +0.329 | 0.145 | +0.409, +0.209, +0.491, +0.145, +0.391 |
| **recency control** | +0.545 | 0.207 | +0.518, +0.218, +0.564, +0.655, +0.773 |

- paired attention − recency: **-0.0745** (t-test p = 0.652)
- paired rollout − recency: **-0.2164** (t-test p = 0.090)

### Where the model actually looks

| patch | hours | occlusion ΔMAE | temporal attention | rollout |
|---|---|---|---|---|
| p0 | 0-16 | **+0.0137** ± 0.0178 | 0.0905 ± 0.0005 | 0.0632 ± 0.0101 |
| p1 | 8-24 | **+0.0140** ± 0.0201 | 0.0904 ± 0.0002 | 0.0547 ± 0.0045 |
| p2 | 16-32 | **+0.0030** ± 0.0181 | 0.0905 ± 0.0004 | 0.0634 ± 0.0026 |
| p3 | 24-40 | **+0.0030** ± 0.0155 | 0.0905 ± 0.0004 | 0.0654 ± 0.0027 |
| p4 | 32-48 | **+0.0012** ± 0.0146 | 0.0907 ± 0.0003 | 0.0537 ± 0.0022 |
| p5 | 40-56 | **-0.0048** ± 0.0133 | 0.0908 ± 0.0003 | 0.0593 ± 0.0019 |
| p6 | 48-64 | **-0.0051** ± 0.0116 | 0.0908 ± 0.0003 | 0.0583 ± 0.0026 |
| p7 | 56-72 | **+0.0449** ± 0.0218 | 0.0911 ± 0.0002 | 0.0507 ± 0.0029 |
| p8 | 64-80 | **+0.0606** ± 0.0502 | 0.0911 ± 0.0001 | 0.0606 ± 0.0024 |
| p9 | 72-88 | **+0.1860** ± 0.0516 | 0.0914 ± 0.0005 | 0.0697 ± 0.0037 |
| p10 | 80-96 | **+2.0611** ± 0.1108 | 0.0920 ± 0.0016 | 0.4009 ± 0.0249 |

Attention spread: **0.0904 … 0.0920**, i.e. 1.7% of its own mean, against a uniform value of 1/11 = 0.0909. Occlusion importance spans -0.0051 … +2.0611.


---

## full (RevIN on, entropy on) — Jena, 5 seeds

### Spearman agreement with the occlusion ranking

| | mean ρ | sd | per seed |
|---|---|---|---|
| temporal attention | +0.225 | 0.293 | +0.373, +0.582, -0.091, -0.064, +0.327 |
| attention rollout | +0.453 | 0.245 | +0.145, +0.391, +0.827, +0.482, +0.418 |
| **recency control** | +0.565 | 0.217 | +0.827, +0.618, +0.391, +0.691, +0.300 |

- paired attention − recency: **-0.3400** (t-test p = 0.082)
- paired rollout − recency: **-0.1127** (t-test p = 0.580)

### Where the model actually looks

| patch | hours | occlusion ΔMAE | temporal attention | rollout |
|---|---|---|---|---|
| p0 | 0-16 | **+0.0226** ± 0.0343 | 0.0911 ± 0.0005 | 0.0884 ± 0.0042 |
| p1 | 8-24 | **+0.0548** ± 0.0500 | 0.0911 ± 0.0005 | 0.0849 ± 0.0047 |
| p2 | 16-32 | **+0.0596** ± 0.0546 | 0.0904 ± 0.0003 | 0.0907 ± 0.0040 |
| p3 | 24-40 | **+0.0506** ± 0.0755 | 0.0909 ± 0.0001 | 0.0804 ± 0.0024 |
| p4 | 32-48 | **+0.0398** ± 0.0652 | 0.0911 ± 0.0002 | 0.0794 ± 0.0028 |
| p5 | 40-56 | **+0.0240** ± 0.0561 | 0.0905 ± 0.0004 | 0.0819 ± 0.0014 |
| p6 | 48-64 | **+0.0329** ± 0.0617 | 0.0910 ± 0.0002 | 0.0798 ± 0.0047 |
| p7 | 56-72 | **+0.0821** ± 0.0715 | 0.0910 ± 0.0001 | 0.0808 ± 0.0020 |
| p8 | 64-80 | **+0.0797** ± 0.0461 | 0.0904 ± 0.0005 | 0.0890 ± 0.0022 |
| p9 | 72-88 | **+0.3239** ± 0.0632 | 0.0909 ± 0.0002 | 0.0878 ± 0.0042 |
| p10 | 80-96 | **+1.8733** ± 0.1534 | 0.0917 ± 0.0002 | 0.1571 ± 0.0167 |

Attention spread: **0.0904 … 0.0917**, i.e. 1.4% of its own mean, against a uniform value of 1/11 = 0.0909. Occlusion importance spans +0.0226 … +1.8733.


---

## no_entropy (RevIN on, entropy off) — Jena, 3 seeds

### Spearman agreement with the occlusion ranking

| | mean ρ | sd | per seed |
|---|---|---|---|
| temporal attention | +0.285 | 0.351 | +0.318, +0.618, -0.082 |
| attention rollout | +0.373 | 0.331 | -0.009, +0.555, +0.573 |
| **recency control** | +0.667 | 0.227 | +0.927, +0.564, +0.509 |

- paired attention − recency: **-0.3818** (t-test p = 0.222)
- paired rollout − recency: **-0.2939** (t-test p = 0.458)

### Where the model actually looks

| patch | hours | occlusion ΔMAE | temporal attention | rollout |
|---|---|---|---|---|
| p0 | 0-16 | **+0.0212** ± 0.0218 | 0.0907 ± 0.0004 | 0.0900 ± 0.0071 |
| p1 | 8-24 | **+0.0395** ± 0.0228 | 0.0906 ± 0.0004 | 0.0824 ± 0.0017 |
| p2 | 16-32 | **+0.0523** ± 0.0351 | 0.0906 ± 0.0001 | 0.0894 ± 0.0053 |
| p3 | 24-40 | **+0.0368** ± 0.0686 | 0.0909 ± 0.0003 | 0.0806 ± 0.0022 |
| p4 | 32-48 | **+0.0254** ± 0.0568 | 0.0910 ± 0.0001 | 0.0774 ± 0.0025 |
| p5 | 40-56 | **+0.0189** ± 0.0590 | 0.0909 ± 0.0001 | 0.0812 ± 0.0027 |
| p6 | 48-64 | **+0.0239** ± 0.0702 | 0.0911 ± 0.0003 | 0.0810 ± 0.0055 |
| p7 | 56-72 | **+0.0833** ± 0.0742 | 0.0909 ± 0.0001 | 0.0798 ± 0.0020 |
| p8 | 64-80 | **+0.0731** ± 0.0471 | 0.0907 ± 0.0002 | 0.0893 ± 0.0010 |
| p9 | 72-88 | **+0.3140** ± 0.0253 | 0.0911 ± 0.0001 | 0.0886 ± 0.0054 |
| p10 | 80-96 | **+1.8904** ± 0.0985 | 0.0915 ± 0.0001 | 0.1602 ± 0.0231 |

Attention spread: **0.0906 … 0.0915**, i.e. 1.0% of its own mean, against a uniform value of 1/11 = 0.0909. Occlusion importance spans +0.0189 … +1.8904.


---

## Headline model (no_revin: RevIN off, entropy on) — Beijing, 5 seeds

### Spearman agreement with the occlusion ranking

| | mean ρ | sd | per seed |
|---|---|---|---|
| temporal attention | +0.622 | 0.394 | +0.155, +0.236, +0.882, +0.991, +0.845 |
| attention rollout | +0.389 | 0.263 | +0.318, +0.745, +0.473, +0.018, +0.391 |
| **recency control** | +0.407 | 0.401 | -0.027, -0.036, +0.709, +0.709, +0.682 |

- paired attention − recency: **+0.2145** (t-test p = 0.001)
- paired rollout − recency: **-0.0182** (t-test p = 0.948)

### Where the model actually looks

| patch | hours | occlusion ΔMAE | temporal attention | rollout |
|---|---|---|---|---|
| p0 | 0-16 | **+0.0089** ± 0.0141 | 0.0907 ± 0.0003 | 0.0757 ± 0.0031 |
| p1 | 8-24 | **-0.0004** ± 0.0194 | 0.0906 ± 0.0001 | 0.0675 ± 0.0037 |
| p2 | 16-32 | **-0.0181** ± 0.0219 | 0.0903 ± 0.0001 | 0.0738 ± 0.0051 |
| p3 | 24-40 | **-0.0172** ± 0.0203 | 0.0904 ± 0.0000 | 0.0733 ± 0.0039 |
| p4 | 32-48 | **-0.0188** ± 0.0263 | 0.0905 ± 0.0001 | 0.0638 ± 0.0033 |
| p5 | 40-56 | **-0.0202** ± 0.0199 | 0.0904 ± 0.0001 | 0.0720 ± 0.0032 |
| p6 | 48-64 | **-0.0078** ± 0.0170 | 0.0908 ± 0.0000 | 0.0674 ± 0.0020 |
| p7 | 56-72 | **+0.0043** ± 0.0457 | 0.0912 ± 0.0002 | 0.0641 ± 0.0022 |
| p8 | 64-80 | **+0.0015** ± 0.0483 | 0.0909 ± 0.0002 | 0.0706 ± 0.0010 |
| p9 | 72-88 | **+0.3243** ± 0.0693 | 0.0917 ± 0.0001 | 0.0764 ± 0.0031 |
| p10 | 80-96 | **+2.0040** ± 0.2383 | 0.0925 ± 0.0004 | 0.2955 ± 0.0174 |

Attention spread: **0.0903 … 0.0925**, i.e. 2.5% of its own mean, against a uniform value of 1/11 = 0.0909. Occlusion importance spans -0.0202 … +2.0040.


---

## no_revin+no_entropy (RevIN off, entropy off) — clean control for the regulariser — Beijing, 5 seeds

### Spearman agreement with the occlusion ranking

| | mean ρ | sd | per seed |
|---|---|---|---|
| temporal attention | +0.567 | 0.248 | +0.209, +0.618, +0.445, +0.718, +0.845 |
| attention rollout | +0.429 | 0.217 | +0.500, +0.418, +0.691, +0.091, +0.445 |
| **recency control** | +0.351 | 0.275 | +0.091, +0.300, +0.291, +0.818, +0.255 |

- paired attention − recency: **+0.2164** (t-test p = 0.133)
- paired rollout − recency: **+0.0782** (t-test p = 0.728)

### Where the model actually looks

| patch | hours | occlusion ΔMAE | temporal attention | rollout |
|---|---|---|---|---|
| p0 | 0-16 | **+0.0098** ± 0.0097 | 0.0906 ± 0.0002 | 0.0768 ± 0.0049 |
| p1 | 8-24 | **+0.0036** ± 0.0184 | 0.0906 ± 0.0001 | 0.0636 ± 0.0029 |
| p2 | 16-32 | **-0.0116** ± 0.0250 | 0.0900 ± 0.0001 | 0.0750 ± 0.0045 |
| p3 | 24-40 | **-0.0189** ± 0.0236 | 0.0903 ± 0.0002 | 0.0722 ± 0.0025 |
| p4 | 32-48 | **-0.0209** ± 0.0206 | 0.0905 ± 0.0002 | 0.0606 ± 0.0021 |
| p5 | 40-56 | **-0.0150** ± 0.0257 | 0.0900 ± 0.0002 | 0.0712 ± 0.0033 |
| p6 | 48-64 | **+0.0013** ± 0.0295 | 0.0907 ± 0.0002 | 0.0654 ± 0.0030 |
| p7 | 56-72 | **-0.0075** ± 0.0276 | 0.0912 ± 0.0001 | 0.0617 ± 0.0034 |
| p8 | 64-80 | **-0.0122** ± 0.0178 | 0.0908 ± 0.0002 | 0.0695 ± 0.0016 |
| p9 | 72-88 | **+0.3572** ± 0.0900 | 0.0919 ± 0.0003 | 0.0748 ± 0.0027 |
| p10 | 80-96 | **+2.3605** ± 0.1912 | 0.0934 ± 0.0005 | 0.3094 ± 0.0227 |

Attention spread: **0.0900 … 0.0934**, i.e. 3.7% of its own mean, against a uniform value of 1/11 = 0.0909. Occlusion importance spans -0.0209 … +2.3605.


---

**`analysis/block_missing.md`**

## Jena

### MAE, mean ± sd over 5 seeds

| Model | intact | A: last 16 h lost | B: first 16 h lost | Δ A | Δ B |
|---|---|---|---|---|---|
| Autoformer | 3.903±0.047 | 5.012±0.112 | 4.053±0.058 | **+28.4%** | +3.8% |
| TimesNet | 3.313±0.015 | 5.005±0.117 | 3.400±0.033 | **+51.1%** | +2.6% |
| iTransformer | 3.300±0.013 | 5.146±0.058 | 3.372±0.020 | **+55.9%** | +2.2% |
| PatchTST | 3.352±0.019 | 5.230±0.087 | 3.434±0.039 | **+56.0%** | +2.4% |
| Crossformer | 3.110±0.020 | 4.920±0.035 | 3.130±0.023 | **+58.2%** | +0.6% |
| Informer | 3.457±0.065 | 5.523±0.123 | 3.602±0.052 | **+59.8%** | +4.2% |
| DLinear | 3.427±0.008 | 5.610±0.020 | 3.493±0.010 | **+63.7%** | +1.9% |
| **MeteoFormer** | 3.028±0.031 | 5.251±0.289 | 3.045±0.030 | **+73.4%** | +0.6% |
| Transformer | 3.452±0.067 | 6.032±0.244 | 3.437±0.056 | **+74.7%** | -0.4% |
| TFT | 3.388±0.066 | 6.046±0.307 | 3.444±0.080 | **+78.5%** | +1.7% |
| LSTM | 3.317±0.045 | 5.976±0.115 | 3.318±0.045 | **+80.1%** | +0.0% |

Across all 55 (model, seed) pairs, losing the **last** 16 h costs **+61.8%** MAE on average, losing the **first** 16 h costs **+1.8%**. Paired Wilcoxon A vs B: p = 1.1e-10. A is worse than B for **55 of 55** pairs.

### Does first place survive an outage in the last 16 h?

| Rank | intact | A: last 16 h lost |
|---|---|---|
| 1 | **MeteoFormer** 3.028 | Crossformer 4.920 |
| 2 | Crossformer 3.110 | TimesNet 5.005 |
| 3 | iTransformer 3.300 | Autoformer 5.012 |
| 4 | TimesNet 3.313 | iTransformer 5.146 |
| 5 | LSTM 3.317 | PatchTST 5.230 |
| 6 | PatchTST 3.352 | **MeteoFormer** 5.251 |
| 7 | TFT 3.388 | Informer 5.523 |
| 8 | DLinear 3.427 | DLinear 5.610 |
| 9 | Transformer 3.452 | LSTM 5.976 |
| 10 | Informer 3.457 | Transformer 6.032 |
| 11 | Autoformer 3.903 | TFT 6.046 |

Under an outage in the last 16 h our model falls from **1st to 6th** by MAE.

### Per target, Δ% of MAE under A (last 16 h lost)

| Model | T | RH | P | WS |
|---|---|---|---|---|
| Autoformer | +27 | +23 | +46 | +11 |
| Crossformer | +47 | +52 | +104 | +23 |
| DLinear | +71 | +70 | +51 | +36 |
| Informer | +37 | +65 | +81 | +28 |
| LSTM | +58 | +92 | +86 | +37 |
| PatchTST | +50 | +50 | +94 | +24 |
| TFT | +69 | +87 | +78 | +36 |
| TimesNet | +46 | +45 | +84 | +25 |
| Transformer | +58 | +89 | +66 | +34 |
| **MeteoFormer** | +73 | +67 | +111 | +28 |
| iTransformer | +48 | +50 | +91 | +29 |


---

## Beijing (Aotizhongxin)

### MAE, mean ± sd over 5 seeds

| Model | intact | A: last 16 h lost | B: first 16 h lost | Δ A | Δ B |
|---|---|---|---|---|---|
| Autoformer | 5.347±0.112 | 6.007±0.094 | 5.493±0.082 | **+12.4%** | +2.7% |
| Transformer | 4.895±0.071 | 6.196±0.129 | 4.938±0.052 | **+26.6%** | +0.9% |
| TimesNet | 4.426±0.033 | 6.110±0.100 | 4.567±0.072 | **+38.1%** | +3.2% |
| Informer | 4.935±0.092 | 7.033±0.050 | 5.172±0.089 | **+42.5%** | +4.8% |
| PatchTST | 4.369±0.046 | 6.394±0.045 | 4.469±0.043 | **+46.3%** | +2.3% |
| iTransformer | 4.250±0.017 | 6.249±0.076 | 4.354±0.013 | **+47.0%** | +2.4% |
| Crossformer | 4.248±0.095 | 6.338±0.178 | 4.295±0.085 | **+49.2%** | +1.1% |
| DLinear | 4.361±0.008 | 6.587±0.034 | 4.487±0.009 | **+51.0%** | +2.9% |
| LSTM | 4.343±0.034 | 6.893±0.116 | 4.432±0.039 | **+58.7%** | +2.0% |
| **MeteoFormer** | 4.150±0.039 | 6.637±0.244 | 4.167±0.040 | **+59.9%** | +0.4% |
| TFT | 4.676±0.177 | 8.070±0.112 | 4.814±0.172 | **+72.7%** | +3.0% |

Across all 55 (model, seed) pairs, losing the **last** 16 h costs **+45.9%** MAE on average, losing the **first** 16 h costs **+2.3%**. Paired Wilcoxon A vs B: p = 1.1e-10. A is worse than B for **55 of 55** pairs.

### Does first place survive an outage in the last 16 h?

| Rank | intact | A: last 16 h lost |
|---|---|---|
| 1 | **MeteoFormer** 4.150 | Autoformer 6.007 |
| 2 | Crossformer 4.248 | TimesNet 6.110 |
| 3 | iTransformer 4.250 | Transformer 6.196 |
| 4 | LSTM 4.343 | iTransformer 6.249 |
| 5 | DLinear 4.361 | Crossformer 6.338 |
| 6 | PatchTST 4.369 | PatchTST 6.394 |
| 7 | TimesNet 4.426 | DLinear 6.587 |
| 8 | TFT 4.676 | **MeteoFormer** 6.637 |
| 9 | Transformer 4.895 | LSTM 6.893 |
| 10 | Informer 4.935 | Informer 7.033 |
| 11 | Autoformer 5.347 | TFT 8.070 |

Under an outage in the last 16 h our model falls from **1st to 8th** by MAE.

### Per target, Δ% of MAE under A (last 16 h lost)

| Model | T | RH | P | WS |
|---|---|---|---|---|
| Autoformer | -1 | +16 | +10 | +7 |
| Crossformer | +30 | +51 | +77 | +17 |
| DLinear | +68 | +52 | +42 | +28 |
| Informer | +21 | +50 | +43 | +21 |
| LSTM | +43 | +63 | +65 | +23 |
| PatchTST | +50 | +44 | +61 | +21 |
| TFT | +74 | +80 | +49 | +31 |
| TimesNet | +19 | +39 | +60 | +15 |
| Transformer | +12 | +27 | +44 | +8 |
| **MeteoFormer** | +63 | +57 | +83 | +19 |
| iTransformer | +43 | +48 | +51 | +21 |


---

**`analysis/missing_robustness.md`**

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
| Crossformer | 4.248±0.095 | 4.256±0.095 | 4.264±0.095 | 4.288±0.095 | +0.95 |

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
| Crossformer | 8.009±0.174 | 8.021±0.177 | 8.037±0.182 | 8.073±0.187 | +0.80 |

### Per-target degradation, Δ% of MAE at 20 % missing

| Model | T | RH | P | WS |
|---|---|---|---|---|
| Autoformer | -0.70 | +0.26 | -0.09 | +0.10 |
| Crossformer | +0.14 | +1.14 | +1.26 | +0.48 |
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
| 1 | **MeteoFormer** | **MeteoFormer** |
| 2 | Crossformer | iTransformer  ←changed |
| 3 | iTransformer | Crossformer  ←changed |
| 4 | LSTM | LSTM |
| 5 | DLinear | DLinear |
| 6 | PatchTST | PatchTST |
| 7 | TimesNet | TimesNet |
| 8 | TFT | TFT |
| 9 | Transformer | Transformer |
| 10 | Informer | Informer |
| 11 | Autoformer | Autoformer |

Ranking by MAE is **changed** between the clean test and 20 % missing.


---

**`analysis/aug_robustness.md`**

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

**`analysis/cross_station.md`**

## Mean over the 11 target stations

Per model: the seed-mean MAE at each station, then mean ± sd across stations; in-domain Aotizhongxin for reference; rank = rank by the cross-station mean MAE.

| Rank | Model | in-domain MAE | transfer MAE (mean ± sd over stations) | transfer RMSE | Δ vs in-domain | stations where 1st |
|---|---|---|---|---|---|---|
| 1 | MeteoFormer | 4.151 | 4.114 ± 0.068 | 8.125 ± 0.163 | -0.036 | 9/11 |
| 2 | iTransformer | 4.251 | 4.183 ± 0.096 | 8.109 ± 0.204 | -0.068 | 2/11 |
| 3 | Crossformer | 4.247 | 4.211 ± 0.076 | 7.876 ± 0.196 | -0.036 | 0/11 |
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
| Changping | iTransformer | 2 / 11 | 0.85 |
| Dingling | iTransformer | 2 / 11 | 0.85 |
| Dongsi | MeteoFormer | 1 / 11 | 0.96 |
| Guanyuan | MeteoFormer | 1 / 11 | 0.96 |
| Gucheng | MeteoFormer | 1 / 11 | 0.89 |
| Huairou | MeteoFormer | 1 / 11 | 0.78 |
| Nongzhanguan | MeteoFormer | 1 / 11 | 1.00 |
| Shunyi | MeteoFormer | 1 / 11 | 0.96 |
| Tiantan | MeteoFormer | 1 / 11 | 1.00 |
| Wanliu | MeteoFormer | 1 / 11 | 0.93 |
| Wanshouxigong | MeteoFormer | 1 / 11 | 0.96 |

Kendall τ over stations: mean 0.92, min 0.78, max 1.00.

Per-channel replication: the error budget behind these aggregates is the same in domain and on the unseen stations — our deficit against Crossformer is the humidity channel and nothing else (in domain RH +21.8, transfer +24.1 in MSE units, while temperature stays at -7.8 / -8.2 in our favour). See the replication section of [analysis/error_decomposition.md](analysis/error_decomposition.md).

**`analysis/cross_station.md`**

## MAE per station (seed mean)

| Model | Changping | Dingling | Dongsi | Guanyuan | Gucheng | Huairou | Nongzhanguan | Shunyi | Tiantan | Wanliu | Wanshouxigong |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MeteoFormer | 4.010 | 4.017 | 4.176 | 4.168 | 4.111 | 4.084 | 4.178 | 4.036 | 4.178 | 4.121 | 4.179 |
| iTransformer | 4.009 | 4.017 | 4.258 | 4.245 | 4.187 | 4.162 | 4.254 | 4.120 | 4.252 | 4.257 | 4.249 |
| Crossformer | 4.086 | 4.103 | 4.261 | 4.257 | 4.239 | 4.263 | 4.250 | 4.091 | 4.245 | 4.268 | 4.255 |
| DLinear | 4.104 | 4.104 | 4.362 | 4.362 | 4.301 | 4.228 | 4.362 | 4.199 | 4.362 | 4.342 | 4.362 |
| PatchTST | 4.114 | 4.114 | 4.370 | 4.370 | 4.349 | 4.359 | 4.370 | 4.255 | 4.370 | 4.417 | 4.370 |
| LSTM | 4.178 | 4.161 | 4.330 | 4.335 | 4.353 | 4.470 | 4.340 | 4.224 | 4.349 | 4.380 | 4.341 |
| TimesNet | 4.196 | 4.209 | 4.408 | 4.425 | 4.365 | 4.412 | 4.422 | 4.267 | 4.418 | 4.421 | 4.420 |
| TFT | 4.559 | 4.573 | 4.639 | 4.671 | 4.677 | 4.830 | 4.676 | 4.560 | 4.722 | 4.705 | 4.706 |
| Transformer | 4.674 | 4.750 | 4.841 | 4.865 | 4.849 | 4.942 | 4.859 | 4.765 | 4.939 | 4.884 | 4.898 |
| Informer | 4.996 | 5.085 | 4.966 | 4.943 | 4.995 | 5.445 | 4.942 | 4.876 | 4.948 | 5.003 | 4.962 |
| Autoformer | 5.249 | 5.222 | 5.353 | 5.371 | 5.288 | 5.277 | 5.362 | 5.195 | 5.392 | 5.309 | 5.391 |

**`analysis/window_sweep.md`**

## Jena

| L (h) | n seeds | val loss | MAE | ΔMAE vs 96 | p (Welch) | DM ΔL1 (p_Holm) | DM ΔL2 (p_Holm) | RMSE | R² |
|---|---|---|---|---|---|---|---|---|---|
| 24 | 5 | 0.1389 | 2.940 ± 0.020 | -0.085 | 0.00143 | -0.0857\* (<1e-15, n=13908) | -1.2340\* (<1e-15, n=13908) | 5.147 ± 0.036 | 0.693 ± 0.001 |
| 48 | 5 | 0.1417 | 2.994 ± 0.021 | -0.031 | 0.111 | -0.0323\* (9.29e-07, n=13908) | -0.6518\* (5.91e-07, n=13908) | 5.205 ± 0.030 | 0.685 ± 0.002 |
| 96 (published) | 5 | 0.1437 | 3.025 ± 0.031 | — | — | — | — | 5.267 ± 0.046 | 0.682 ± 0.005 |
| 192 | 5 | 0.1457 | 3.041 ± 0.023 | +0.016 | 0.39 | +0.0292\* (0.00265, n=13812) | +0.4469\* (0.0165, n=13812) | 5.272 ± 0.052 | 0.680 ± 0.003 |

## Beijing (Aotizhongxin)

| L (h) | n seeds | val loss | MAE | ΔMAE vs 96 | p (Welch) | DM ΔL1 (p_Holm) | DM ΔL2 (p_Holm) | RMSE | R² |
|---|---|---|---|---|---|---|---|---|---|
| 24 | 5 | 0.1580 | 4.058 ± 0.042 | -0.093 | 0.00688 | -0.1166\* (3.36e-07, n=6895) | -3.3760\* (0.000547, n=6895) | 8.067 ± 0.092 | 0.664 ± 0.004 |
| 48 | 5 | 0.1590 | 4.107 ± 0.090 | -0.044 | 0.358 | -0.0503\* (0.0098, n=6895) | -2.3414\* (0.00376, n=6895) | 8.081 ± 0.083 | 0.662 ± 0.006 |
| 96 (published) | 5 | 0.1592 | 4.151 ± 0.039 | — | — | — | — | 8.216 ± 0.049 | 0.659 ± 0.002 |
| 192 | 5 | 0.1648 | 4.234 ± 0.059 | +0.083 | 0.0333 | +0.0667\* (0.0098, n=6799) | +1.9056 (0.0736, n=6799) | 8.365 ± 0.191 | 0.652 ± 0.006 |

## Selection on validation

- Jena: lowest mean validation loss at **L = 24** (0.1389; L = 96: 0.1437).
- Beijing (Aotizhongxin): lowest mean validation loss at **L = 24** (0.1580; L = 96: 0.1592).

**`analysis/variance_calibration.md`**

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

**`analysis/loss_mse.md`** — appendix: MSE instead of Huber

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
| Crossformer | 4.247 | 4.110 | -0.137 | 2 | 1 |
| **MeteoFormer** | 4.151 | 4.224 | +0.073 | 1 | 2 |
| iTransformer | 4.251 | 4.251 | -0.000 | 3 | 3 |
| TimesNet | 4.426 | 4.294 | -0.132 | 7 | 4 |
| LSTM | 4.343 | 4.300 | -0.043 | 4 | 5 |
| PatchTST | 4.370 | 4.347 | -0.023 | 6 | 6 |
| DLinear | 4.362 | 4.416 | +0.054 | 5 | 7 |
| TFT | 4.676 | 4.685 | +0.009 | 8 | 8 |
| Informer | 4.932 | 4.733 | -0.199 | 10 | 9 |
| Transformer | 4.895 | 4.779 | -0.115 | 9 | 10 |
| Autoformer | 5.347 | 5.257 | -0.090 | 11 | 11 |

### RMSE

| Model | RMSE published | RMSE variant | Δ | rank published | rank variant |
|---|---|---|---|---|---|
| Crossformer | 8.002 | 7.938 | -0.064 | 1 | 1 |
| LSTM | 8.103 | 8.040 | -0.063 | 2 | 2 |
| TimesNet | 8.273 | 8.164 | -0.109 | 5 | 3 |
| **MeteoFormer** | 8.216 | 8.185 | -0.031 | 3 | 4 |
| iTransformer | 8.266 | 8.214 | -0.052 | 4 | 5 |
| PatchTST | 8.336 | 8.340 | +0.004 | 6 | 6 |
| DLinear | 8.381 | 8.369 | -0.012 | 7 | 7 |
| Informer | 8.965 | 8.698 | -0.267 | 10 | 8 |
| TFT | 8.431 | 8.801 | +0.370 | 8 | 9 |
| Transformer | 8.923 | 8.895 | -0.028 | 9 | 10 |
| Autoformer | 9.417 | 9.434 | +0.018 | 11 | 11 |

### Why it falls short: dispersion of the humidity forecast

`sd ratio RH` = sd of the predicted RH over sd of the observed RH. A squared loss is minimised below 1 (the oracle column, fitted on test, is a measurement of the over-dispersion, not a method).

| Variant | sd ratio RH | oracle a (RH) |
|---|---|---|
| published (Huber) | 0.933 | 0.819 |
| with MSE | 0.919 | 0.838 |
| Crossformer (published) | 0.849 | 0.916 |

## 7. Attention stability (for the reconstructed Table 6)

**`analysis/attention_stability.md`**

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
