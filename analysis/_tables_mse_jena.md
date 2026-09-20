# Error decomposition on Jena: where the RMSE gap lives

Diagnostic only. Nothing retrained, nothing tuned; the test set is read, never fitted. Numbers are mean±sd over 5 seeds, in physical units (T °C, RH %, P mbar, WS m/s). Aggregate rows average the four targets, which is how the paper's headline MAE/RMSE are defined, so they are dominated by whichever channel has the largest scale.

## 0. Overall (reproduced from the dumped arrays)

| Variant | MAE | RMSE |
|---|---|---|
| Ours (MSE) | 3.023±0.027 | 5.214±0.059 |
| Ours (Huber, headline) | 3.025±0.031 | 5.267±0.046 |
| Crossformer | 3.107±0.018 | 5.166±0.021 |

## 1. Per target channel

| Variant | metric | T | RH | P | WS |
|---|---|---|---|---|---|
| Ours (MSE) | MAE | 1.912±0.056 | 7.118±0.097 | 2.188±0.062 | 0.874±0.011 |
| Ours (MSE) | RMSE | 2.527±0.079 | 9.526±0.131 | 3.181±0.055 | 1.216±0.013 |
| Ours (Huber, headline) | MAE | 1.953±0.033 | 7.152±0.064 | 2.145±0.046 | 0.849±0.003 |
| Ours (Huber, headline) | RMSE | 2.547±0.025 | 9.651±0.092 | 3.138±0.047 | 1.215±0.009 |
| Crossformer | MAE | 2.531±0.103 | 6.927±0.061 | 2.138±0.063 | 0.833±0.008 |
| Crossformer | RMSE | 3.162±0.132 | 9.254±0.036 | 3.107±0.053 | 1.207±0.007 |

## 2. Per horizon (all 4 targets averaged)

| Variant | metric | h=1 | h=6 | h=12 | h=24 |
|---|---|---|---|---|---|
| Ours (MSE) | MAE | 1.247±0.126 | 2.505±0.042 | 3.189±0.049 | 3.874±0.026 |
| Ours (MSE) | RMSE | 2.172±0.241 | 4.564±0.097 | 5.440±0.111 | 6.147±0.024 |
| Ours (Huber, headline) | MAE | 1.208±0.095 | 2.530±0.045 | 3.200±0.040 | 3.864±0.043 |
| Ours (Huber, headline) | RMSE | 2.091±0.157 | 4.625±0.107 | 5.502±0.070 | 6.214±0.069 |
| Crossformer | MAE | 1.492±0.076 | 2.639±0.039 | 3.226±0.026 | 3.918±0.018 |
| Crossformer | RMSE | 2.297±0.057 | 4.496±0.036 | 5.352±0.076 | 6.136±0.024 |

### 2b. Per horizon, temperature only

| Variant | metric | h=1 | h=6 | h=12 | h=24 |
|---|---|---|---|---|---|
| Ours (MSE) | MAE | 0.877±0.113 | 1.617±0.049 | 2.022±0.018 | 2.416±0.118 |
| Ours (MSE) | RMSE | 1.156±0.165 | 2.111±0.067 | 2.604±0.030 | 3.097±0.161 |
| Ours (Huber, headline) | MAE | 0.906±0.097 | 1.667±0.056 | 2.094±0.043 | 2.431±0.055 |
| Ours (Huber, headline) | RMSE | 1.149±0.123 | 2.148±0.073 | 2.665±0.043 | 3.094±0.057 |
| Crossformer | MAE | 2.083±0.226 | 2.435±0.206 | 2.481±0.068 | 2.783±0.047 |
| Crossformer | RMSE | 2.459±0.186 | 2.987±0.217 | 3.138±0.126 | 3.508±0.089 |

## 3. Amplitude deciles

Each window gets, per channel, the excursion `max_h |y_true[h] - y_last_input|`; windows are ranked into deciles **within each channel**, so D1 = flattest 10 % of windows for that channel and D10 = the most volatile 10 %. Errors are then pooled over the 4 channels within a decile.

### 3a. Decile definition (amplitude range, ground truth only)

| decile | T | RH | P | WS |
|---|---|---|---|---|
| D1 | 0.38–2.50 | 1.45–11.84 | 0.36–1.53 | 0.25–1.12 |
| D2 | 2.50–3.51 | 11.84–16.02 | 1.53–2.11 | 1.12–1.48 |
| D3 | 3.51–4.41 | 16.02–19.19 | 2.11–2.73 | 1.48–1.79 |
| D4 | 4.41–5.23 | 19.19–22.39 | 2.73–3.35 | 1.79–2.09 |
| D5 | 5.23–6.10 | 22.39–25.49 | 3.35–4.06 | 2.09–2.41 |
| D6 | 6.10–7.00 | 25.49–28.81 | 4.06–4.94 | 2.41–2.78 |
| D7 | 7.00–8.07 | 28.81–32.51 | 4.94–5.92 | 2.78–3.21 |
| D8 | 8.07–9.40 | 32.51–36.99 | 5.92–7.16 | 3.21–3.79 |
| D9 | 9.40–11.36 | 36.99–42.96 | 7.16–9.24 | 3.79–4.63 |
| D10 | 11.36–21.26 | 42.96–67.99 | 9.24–28.21 | 4.63–10.23 |

### 3b. MAE by amplitude decile (all targets pooled)

| Variant | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours (MSE) | 2.510±0.110 | 2.569±0.088 | 2.629±0.070 | 2.761±0.060 | 2.850±0.045 | 2.937±0.031 | 3.041±0.008 | 3.193±0.029 | 3.449±0.044 | 4.289±0.163 |
| Ours (Huber, headline) | 2.393±0.085 | 2.498±0.056 | 2.596±0.068 | 2.756±0.053 | 2.863±0.054 | 2.966±0.043 | 3.071±0.025 | 3.245±0.055 | 3.488±0.066 | 4.371±0.139 |
| Crossformer | 2.529±0.122 | 2.668±0.104 | 2.724±0.078 | 2.825±0.052 | 2.924±0.034 | 3.030±0.033 | 3.121±0.016 | 3.286±0.041 | 3.550±0.063 | 4.416±0.120 |
| **Ours (MSE) − Ours (Huber, headline)** | +0.117 | +0.070 | +0.033 | +0.005 | -0.013 | -0.029 | -0.030 | -0.053 | -0.039 | -0.082 |
| **Ours (MSE) − Crossformer** | -0.018 | -0.099 | -0.095 | -0.063 | -0.074 | -0.093 | -0.080 | -0.094 | -0.101 | -0.127 |

### 3b. RMSE by amplitude decile (all targets pooled)

| Variant | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours (MSE) | 4.690±0.266 | 4.615±0.196 | 4.699±0.161 | 4.875±0.125 | 4.949±0.123 | 5.041±0.083 | 5.207±0.054 | 5.305±0.021 | 5.629±0.025 | 6.757±0.204 |
| Ours (Huber, headline) | 4.508±0.166 | 4.562±0.141 | 4.701±0.157 | 4.926±0.122 | 5.028±0.090 | 5.130±0.081 | 5.287±0.027 | 5.401±0.066 | 5.721±0.085 | 6.952±0.208 |
| Crossformer | 4.302±0.231 | 4.478±0.184 | 4.588±0.131 | 4.762±0.071 | 4.888±0.032 | 5.031±0.050 | 5.189±0.025 | 5.318±0.070 | 5.697±0.081 | 6.900±0.186 |
| **Ours (MSE) − Ours (Huber, headline)** | +0.182 | +0.053 | -0.001 | -0.051 | -0.078 | -0.089 | -0.080 | -0.096 | -0.092 | -0.196 |
| **Ours (MSE) − Crossformer** | +0.388 | +0.138 | +0.111 | +0.113 | +0.062 | +0.010 | +0.018 | -0.013 | -0.068 | -0.143 |

### 3c. Share of total squared error contributed by each decile

RMSE is a sum of squares, so a decile matters in proportion to the squared error it carries, not to its width. Share = (sum of squared errors in the decile) / (total), %.

| Variant | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours (MSE) | 8.1% | 7.8% | 8.1% | 8.7% | 9.0% | 9.4% | 10.0% | 10.4% | 11.7% | 16.8% |
| Ours (Huber, headline) | 7.3% | 7.5% | 8.0% | 8.7% | 9.1% | 9.5% | 10.1% | 10.5% | 11.8% | 17.5% |
| Crossformer | 6.9% | 7.5% | 7.9% | 8.5% | 8.9% | 9.5% | 10.1% | 10.6% | 12.2% | 17.9% |

### 3d. Temperature only, by its own amplitude decile

| Variant | metric | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours (MSE) | MAE | 1.612±0.098 | 1.674±0.074 | 1.750±0.045 | 1.869±0.030 | 1.920±0.044 | 1.930±0.042 | 1.984±0.076 | 2.024±0.062 | 2.074±0.093 | 2.284±0.213 |
| Ours (MSE) | RMSE | 2.065±0.110 | 2.200±0.097 | 2.285±0.053 | 2.417±0.037 | 2.518±0.054 | 2.506±0.071 | 2.614±0.120 | 2.688±0.104 | 2.784±0.142 | 3.032±0.230 |
| Ours (Huber, headline) | MAE | 1.684±0.143 | 1.721±0.097 | 1.786±0.063 | 1.911±0.055 | 1.940±0.065 | 1.957±0.051 | 2.009±0.040 | 2.050±0.051 | 2.129±0.035 | 2.348±0.116 |
| Ours (Huber, headline) | RMSE | 2.130±0.165 | 2.227±0.108 | 2.308±0.078 | 2.453±0.071 | 2.517±0.093 | 2.513±0.068 | 2.609±0.042 | 2.682±0.061 | 2.801±0.048 | 3.079±0.141 |
| Crossformer | MAE | 2.640±0.160 | 2.588±0.202 | 2.536±0.172 | 2.494±0.168 | 2.516±0.138 | 2.448±0.110 | 2.469±0.096 | 2.522±0.077 | 2.509±0.057 | 2.583±0.080 |
| Crossformer | RMSE | 3.159±0.205 | 3.104±0.199 | 3.079±0.190 | 3.064±0.189 | 3.142±0.179 | 3.077±0.149 | 3.161±0.160 | 3.242±0.115 | 3.294±0.080 | 3.280±0.070 |

## 4. Tail of the error distribution

Absolute-error quantiles over all (window, horizon, channel) triples, and the share of total squared error carried by the worst 1 % of them.

| Variant | p50 | p90 | p99 | p99.9 | max | SE share of worst 1% |
|---|---|---|---|---|---|---|
| Ours (MSE) | 1.478±0.026 | 7.723±0.071 | 20.735±0.304 | 34.531±0.633 | 57.401±3.134 | 27.093±0.374 |
| Ours (Huber, headline) | 1.465±0.035 | 7.665±0.069 | 21.112±0.288 | 35.464±1.210 | 56.126±1.143 | 27.828±1.011 |
| Crossformer | 1.647±0.043 | 7.659±0.071 | 20.137±0.145 | 33.672±1.115 | 53.482±1.935 | 26.009±1.079 |

## 5. Where the aggregate gap actually comes from: the channel budget

The headline MAE/RMSE are unweighted means over the four targets in physical units. RH has ~10x the error scale of T, so the aggregate is an RH statistic wearing a coat. Aggregate MSE = mean of the four channel MSEs, so each channel moves it by (its MSE gap)/4.

| Variant | MSE T | MSE RH | MSE P | MSE WS | aggregate MSE | aggregate RMSE |
|---|---|---|---|---|---|---|
| Ours (MSE) | 6.390 | 90.750 | 10.119 | 1.478 | 27.184 | 5.214 |
| Ours (Huber, headline) | 6.490 | 93.151 | 9.849 | 1.476 | 27.741 | 5.267 |
| Crossformer | 10.015 | 85.634 | 9.653 | 1.458 | 26.690 | 5.166 |
| **Ours (MSE) − Crossformer** | -3.625 | +5.116 | +0.465 | +0.021 | +0.494 | — |
| **contribution to aggregate MSE gap** | -0.906 | +1.279 | +0.116 | +0.005 | +0.494 | — |

### 5b. RH only, by RH amplitude decile (RMSE)

| Variant | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours (MSE) | 8.948±0.549 | 8.715±0.396 | 8.838±0.325 | 9.147±0.249 | 9.243±0.254 | 9.352±0.174 | 9.599±0.140 | 9.611±0.092 | 9.955±0.061 | 11.509±0.373 |
| Ours (Huber, headline) | 8.585±0.335 | 8.641±0.288 | 8.865±0.309 | 9.273±0.242 | 9.434±0.163 | 9.572±0.161 | 9.790±0.043 | 9.822±0.134 | 10.151±0.185 | 11.918±0.444 |
| Crossformer | 7.819±0.438 | 8.168±0.338 | 8.383±0.222 | 8.725±0.090 | 8.935±0.046 | 9.181±0.146 | 9.410±0.103 | 9.479±0.190 | 9.986±0.216 | 11.802±0.446 |
| **Ours (MSE) − Crossformer** | +1.129 | +0.547 | +0.455 | +0.422 | +0.308 | +0.171 | +0.189 | +0.133 | -0.031 | -0.293 |

## 6. Is the loss the culprit? (Huber operating regime)

Training uses `HuberLoss(delta=1.0)` on globally standardised targets (train sd: T=8.64, RH=16.46, P=8.29, WS=1.45). Fraction of test residuals past delta, i.e. the share the loss treats as L1 rather than L2:

| Variant | overall | T | RH | P | WS |
|---|---|---|---|---|---|
| Ours (MSE) | 0.072 | 0.005 | 0.085 | 0.028 | 0.171 |
| Ours (Huber, headline) | 0.070 | 0.004 | 0.089 | 0.027 | 0.159 |
| Crossformer | 0.065 | 0.009 | 0.078 | 0.026 | 0.149 |

About 93 % of residuals sit in the quadratic region, so the objective is effectively MSE for every model here. The loss does not explain an MAE-good / RMSE-bad profile.

## 7. What does explain it: RH dispersion

Ratio of predicted sd to ground-truth sd per channel, and the scalar `a` that would minimise RH MSE under `pred -> mean + a*(pred - mean)`. `a < 1` means the forecast is over-dispersed for a squared loss.

| Variant | sd ratio T | sd ratio RH | sd ratio P | sd ratio WS | oracle a (RH) |
|---|---|---|---|---|---|
| Ours (MSE) | 1.032 | 0.927 | 0.963 | 0.666 | 0.884 |
| Ours (Huber, headline) | 0.996 | 0.951 | 0.964 | 0.615 | 0.861 |
| Crossformer | 0.991 | 0.850 | 0.885 | 0.556 | 0.972 |

Effect of applying that RH rescale (oracle, fitted **on test** — a measurement of how much of the gap is dispersion, **not** a proposed method; an honest version must fit `a` on validation):

| Variant | aggregate RMSE as-is | with RH rescaled |
|---|---|---|
| Ours (MSE) | 5.214±0.059 | 5.133±0.037 |
| Ours (Huber, headline) | 5.267±0.046 | 5.146±0.034 |
| Crossformer | 5.166±0.021 | 5.158±0.017 |
