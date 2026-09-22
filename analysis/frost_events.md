# Frost-event classification (Reviewer 2)

Produced by [analysis/frost_events.py](analysis/frost_events.py).

**Adaptation of the regression baselines.** No baseline has an event head. Every model, ours included, is scored from its predicted temperature: the ranking score for AUC/AP is `-T_pred`, the hard decision is `T_pred <= tau`. Scoring our dedicated head against the baselines' thresholded regression would compare two different amounts of supervision, not two models, so the head is evaluated separately in the ablation instead.

**Threshold.** Two variants, side by side. (a) `tau = 0 °C`, the physical freezing point — nothing is fitted, which is the cleanest possible claim. (b) `tau` maximising F1 on the **validation** split per (model, seed), then frozen and applied to test. The test set is never used to pick `tau`.

**Imbalance.** Nothing is resampled. The positive rate is stated below; AP and F1 are reported next to AUC because AUC is optimistic under imbalance. During training our event head used `BCEWithLogitsLoss(pos_weight=(1-r)/r)` with `r` the frost rate of the training split; the baselines have no event loss at all.


---

## Jena

Test positive rate: **0.0722** (24093 of 333792 (window, step) pairs).
Validation positive rate: 0.1075.

### (a) Physical threshold τ = 0 °C (nothing fitted)

| Model | AUC | AP | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Crossformer | 0.980±0.001 | 0.786±0.009 | 0.717±0.018 | 0.688±0.077 | 0.764±0.091 |
| LSTM | 0.976±0.001 | 0.760±0.007 | 0.705±0.007 | 0.696±0.021 | 0.716±0.018 |
| MeteoFormer | 0.975±0.002 | 0.757±0.010 | 0.701±0.009 | 0.667±0.037 | 0.741±0.028 |
| TimesNet | 0.975±0.001 | 0.755±0.010 | 0.702±0.006 | 0.680±0.029 | 0.726±0.022 |
| Transformer | 0.976±0.001 | 0.753±0.017 | 0.706±0.012 | 0.633±0.018 | 0.798±0.038 |
| Informer | 0.974±0.001 | 0.740±0.005 | 0.692±0.019 | 0.606±0.043 | 0.810±0.032 |
| TFT | 0.971±0.005 | 0.729±0.036 | 0.675±0.024 | 0.669±0.051 | 0.688±0.060 |
| PatchTST | 0.969±0.001 | 0.710±0.011 | 0.619±0.020 | 0.730±0.007 | 0.537±0.028 |
| iTransformer | 0.969±0.001 | 0.708±0.005 | 0.640±0.008 | 0.706±0.012 | 0.586±0.020 |
| DLinear | 0.965±0.000 | 0.694±0.001 | 0.553±0.008 | 0.763±0.010 | 0.434±0.013 |
| Autoformer | 0.963±0.001 | 0.649±0.021 | 0.601±0.018 | 0.612±0.025 | 0.590±0.023 |

### (b) τ selected on validation to maximise F1

| Model | τ (°C) | AUC | AP | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| Crossformer | -0.32±2.11 | 0.980±0.001 | 0.786±0.009 | 0.727±0.011 | 0.700±0.018 | 0.758±0.040 |
| LSTM | 0.90±0.34 | 0.976±0.001 | 0.760±0.007 | 0.706±0.008 | 0.651±0.020 | 0.772±0.018 |
| MeteoFormer | -0.29±0.37 | 0.975±0.002 | 0.757±0.010 | 0.704±0.007 | 0.698±0.022 | 0.712±0.023 |
| TimesNet | 0.21±0.23 | 0.975±0.001 | 0.755±0.010 | 0.702±0.007 | 0.665±0.021 | 0.744±0.018 |
| Transformer | -1.00±0.80 | 0.976±0.001 | 0.753±0.017 | 0.709±0.014 | 0.680±0.026 | 0.742±0.013 |
| Informer | -1.02±0.79 | 0.974±0.001 | 0.740±0.005 | 0.700±0.010 | 0.646±0.017 | 0.763±0.011 |
| TFT | 0.43±0.47 | 0.971±0.005 | 0.729±0.036 | 0.682±0.024 | 0.641±0.031 | 0.730±0.022 |
| PatchTST | 0.87±0.18 | 0.969±0.001 | 0.710±0.011 | 0.667±0.007 | 0.669±0.015 | 0.665±0.020 |
| iTransformer | 0.57±0.08 | 0.969±0.001 | 0.708±0.005 | 0.667±0.007 | 0.660±0.014 | 0.675±0.025 |
| DLinear | 1.43±0.08 | 0.965±0.000 | 0.694±0.001 | 0.639±0.001 | 0.623±0.004 | 0.657±0.005 |
| Autoformer | 0.98±0.13 | 0.963±0.001 | 0.649±0.021 | 0.628±0.015 | 0.568±0.024 | 0.703±0.020 |

AUC and AP are threshold-free and therefore identical between (a) and (b); only F1/Precision/Recall move.


---

## Beijing (Aotizhongxin)

Test positive rate: **0.1737** (28752 of 165480 (window, step) pairs).
Validation positive rate: 0.3245.

### (a) Physical threshold τ = 0 °C (nothing fitted)

| Model | AUC | AP | F1 | Precision | Recall |
|---|---|---|---|---|---|
| MeteoFormer | 0.973±0.002 | 0.872±0.007 | 0.793±0.018 | 0.776±0.024 | 0.814±0.060 |
| iTransformer | 0.973±0.000 | 0.870±0.002 | 0.762±0.006 | 0.835±0.003 | 0.700±0.011 |
| Informer | 0.972±0.001 | 0.869±0.007 | 0.793±0.002 | 0.729±0.024 | 0.872±0.038 |
| DLinear | 0.972±0.001 | 0.867±0.002 | 0.714±0.023 | 0.869±0.010 | 0.607±0.037 |
| Crossformer | 0.973±0.002 | 0.862±0.012 | 0.782±0.024 | 0.804±0.027 | 0.765±0.062 |
| PatchTST | 0.970±0.001 | 0.855±0.009 | 0.748±0.006 | 0.834±0.006 | 0.679±0.008 |
| TimesNet | 0.972±0.001 | 0.855±0.008 | 0.802±0.006 | 0.775±0.015 | 0.832±0.020 |
| Transformer | 0.970±0.001 | 0.852±0.006 | 0.771±0.015 | 0.801±0.024 | 0.745±0.050 |
| LSTM | 0.970±0.002 | 0.847±0.009 | 0.770±0.010 | 0.803±0.021 | 0.742±0.037 |
| TFT | 0.968±0.003 | 0.845±0.015 | 0.784±0.012 | 0.769±0.022 | 0.801±0.029 |
| Autoformer | 0.961±0.003 | 0.815±0.011 | 0.753±0.008 | 0.766±0.009 | 0.741±0.009 |

### (b) τ selected on validation to maximise F1

| Model | τ (°C) | AUC | AP | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| MeteoFormer | 0.00±0.00 | 0.973±0.002 | 0.872±0.007 | 0.793±0.018 | 0.776±0.024 | 0.814±0.060 |
| iTransformer | 0.00±0.00 | 0.973±0.000 | 0.870±0.002 | 0.762±0.006 | 0.835±0.003 | 0.700±0.011 |
| Informer | 0.00±0.00 | 0.972±0.001 | 0.869±0.007 | 0.793±0.002 | 0.729±0.024 | 0.872±0.038 |
| DLinear | 0.00±0.00 | 0.972±0.001 | 0.867±0.002 | 0.714±0.023 | 0.869±0.010 | 0.607±0.037 |
| Crossformer | 0.00±0.00 | 0.973±0.002 | 0.862±0.012 | 0.782±0.024 | 0.804±0.027 | 0.765±0.062 |
| PatchTST | 0.00±0.00 | 0.970±0.001 | 0.855±0.009 | 0.748±0.006 | 0.834±0.006 | 0.679±0.008 |
| TimesNet | 0.00±0.00 | 0.972±0.001 | 0.855±0.008 | 0.802±0.006 | 0.775±0.015 | 0.832±0.020 |
| Transformer | 0.00±0.00 | 0.970±0.001 | 0.852±0.006 | 0.771±0.015 | 0.801±0.024 | 0.745±0.050 |
| LSTM | 0.00±0.00 | 0.970±0.002 | 0.847±0.009 | 0.770±0.010 | 0.803±0.021 | 0.742±0.037 |
| TFT | 0.00±0.00 | 0.968±0.003 | 0.845±0.015 | 0.784±0.012 | 0.769±0.022 | 0.801±0.029 |
| Autoformer | 0.00±0.00 | 0.961±0.003 | 0.815±0.011 | 0.753±0.008 | 0.766±0.009 | 0.741±0.009 |

AUC and AP are threshold-free and therefore identical between (a) and (b); only F1/Precision/Recall move.
