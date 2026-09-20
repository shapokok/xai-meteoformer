# Error decomposition on Jena: where the RMSE gap lives

Diagnostic only. Nothing retrained, nothing tuned; the test set is read, never fitted. Numbers are mean±sd over 5 seeds, in physical units (T °C, RH %, P mbar, WS m/s). Aggregate rows average the four targets, which is how the paper's headline MAE/RMSE are defined, so they are dominated by whichever channel has the largest scale.

## 0. Overall (reproduced from the dumped arrays)

| Variant | MAE | RMSE |
|---|---|---|
| Ours (MSE) | 4.224±0.080 | 8.185±0.174 |
| Ours (Huber, headline) | 4.151±0.039 | 8.216±0.049 |
| Crossformer | 4.247±0.095 | 8.002±0.172 |

## 1. Per target channel

| Variant | metric | T | RH | P | WS |
|---|---|---|---|---|---|
| Ours (MSE) | MAE | 2.041±0.041 | 11.739±0.299 | 2.428±0.215 | 0.687±0.004 |
| Ours (MSE) | RMSE | 2.646±0.061 | 15.787±0.371 | 3.292±0.164 | 0.948±0.003 |
| Ours (Huber, headline) | MAE | 2.019±0.048 | 11.668±0.107 | 2.249±0.024 | 0.667±0.006 |
| Ours (Huber, headline) | RMSE | 2.624±0.050 | 15.887±0.099 | 3.132±0.030 | 0.946±0.009 |
| Crossformer | MAE | 3.043±0.213 | 11.088±0.243 | 2.207±0.051 | 0.650±0.003 |
| Crossformer | RMSE | 3.834±0.233 | 15.187±0.312 | 3.141±0.060 | 0.941±0.011 |

## 2. Per horizon (all 4 targets averaged)

| Variant | metric | h=1 | h=6 | h=12 | h=24 |
|---|---|---|---|---|---|
| Ours (MSE) | MAE | 2.019±0.157 | 3.565±0.062 | 4.495±0.046 | 5.222±0.143 |
| Ours (MSE) | RMSE | 3.591±0.190 | 6.896±0.209 | 8.621±0.142 | 9.652±0.272 |
| Ours (Huber, headline) | MAE | 1.842±0.133 | 3.448±0.070 | 4.350±0.054 | 5.207±0.072 |
| Ours (Huber, headline) | RMSE | 3.487±0.159 | 6.824±0.097 | 8.478±0.141 | 9.788±0.139 |
| Crossformer | MAE | 2.125±0.125 | 3.615±0.140 | 4.435±0.082 | 5.281±0.078 |
| Crossformer | RMSE | 3.563±0.143 | 6.744±0.193 | 8.340±0.171 | 9.554±0.256 |

### 2b. Per horizon, temperature only

| Variant | metric | h=1 | h=6 | h=12 | h=24 |
|---|---|---|---|---|---|
| Ours (MSE) | MAE | 1.367±0.151 | 1.898±0.110 | 2.215±0.128 | 2.231±0.033 |
| Ours (MSE) | RMSE | 1.709±0.149 | 2.430±0.097 | 2.814±0.119 | 2.894±0.057 |
| Ours (Huber, headline) | MAE | 1.072±0.086 | 1.906±0.131 | 2.152±0.070 | 2.207±0.044 |
| Ours (Huber, headline) | RMSE | 1.381±0.099 | 2.442±0.169 | 2.773±0.082 | 2.827±0.050 |
| Crossformer | MAE | 2.832±0.485 | 3.008±0.336 | 3.042±0.137 | 3.184±0.083 |
| Crossformer | RMSE | 3.445±0.493 | 3.748±0.369 | 3.866±0.171 | 4.049±0.094 |

## 3. Amplitude deciles

Each window gets, per channel, the excursion `max_h |y_true[h] - y_last_input|`; windows are ranked into deciles **within each channel**, so D1 = flattest 10 % of windows for that channel and D10 = the most volatile 10 %. Errors are then pooled over the 4 channels within a decile.

### 3a. Decile definition (amplitude range, ground truth only)

| decile | T | RH | P | WS |
|---|---|---|---|---|
| D1 | 0.50–3.80 | 1.78–16.02 | 0.60–2.10 | 0.40–1.10 |
| D2 | 3.80–4.90 | 16.02–21.14 | 2.10–2.60 | 1.10–1.30 |
| D3 | 4.90–5.80 | 21.14–25.70 | 2.60–3.10 | 1.30–1.50 |
| D4 | 5.80–6.50 | 25.70–29.29 | 3.10–3.69 | 1.50–1.70 |
| D5 | 6.50–7.20 | 29.29–33.15 | 3.69–4.20 | 1.70–2.00 |
| D6 | 7.20–8.00 | 33.15–36.95 | 4.20–4.90 | 2.00–2.20 |
| D7 | 8.00–8.90 | 36.95–41.17 | 4.90–5.80 | 2.20–2.60 |
| D8 | 8.90–9.90 | 41.17–46.32 | 5.80–7.00 | 2.60–3.00 |
| D9 | 9.90–11.50 | 46.32–54.09 | 7.00–9.00 | 3.00–3.80 |
| D10 | 11.50–18.60 | 54.09–82.06 | 9.00–27.90 | 3.80–8.90 |

### 3b. MAE by amplitude decile (all targets pooled)

| Variant | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours (MSE) | 3.947±0.187 | 3.530±0.146 | 3.467±0.117 | 3.552±0.130 | 3.752±0.097 | 3.848±0.108 | 3.951±0.101 | 4.165±0.060 | 4.995±0.058 | 6.985±0.146 |
| Ours (Huber, headline) | 3.862±0.052 | 3.474±0.049 | 3.372±0.053 | 3.430±0.069 | 3.649±0.054 | 3.799±0.054 | 3.858±0.041 | 4.095±0.053 | 4.923±0.090 | 6.997±0.141 |
| Crossformer | 3.778±0.213 | 3.449±0.175 | 3.505±0.113 | 3.602±0.121 | 3.824±0.093 | 3.917±0.081 | 4.011±0.098 | 4.218±0.112 | 5.092±0.137 | 7.022±0.248 |
| **Ours (MSE) − Ours (Huber, headline)** | +0.085 | +0.057 | +0.095 | +0.121 | +0.103 | +0.049 | +0.093 | +0.070 | +0.072 | -0.012 |
| **Ours (MSE) − Crossformer** | +0.169 | +0.081 | -0.038 | -0.050 | -0.072 | -0.069 | -0.061 | -0.053 | -0.097 | -0.037 |

### 3b. RMSE by amplitude decile (all targets pooled)

| Variant | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours (MSE) | 7.307±0.272 | 6.773±0.204 | 6.732±0.179 | 6.789±0.241 | 7.206±0.191 | 7.170±0.216 | 7.617±0.268 | 7.830±0.229 | 9.285±0.163 | 12.977±0.338 |
| Ours (Huber, headline) | 7.552±0.258 | 6.985±0.159 | 6.708±0.123 | 6.704±0.076 | 7.135±0.094 | 7.213±0.074 | 7.523±0.082 | 7.728±0.077 | 9.222±0.097 | 13.155±0.252 |
| Crossformer | 6.810±0.479 | 6.358±0.422 | 6.529±0.267 | 6.699±0.265 | 6.986±0.168 | 7.010±0.123 | 7.481±0.202 | 7.621±0.254 | 9.212±0.321 | 12.905±0.548 |
| **Ours (MSE) − Ours (Huber, headline)** | -0.244 | -0.211 | +0.024 | +0.085 | +0.070 | -0.043 | +0.095 | +0.103 | +0.063 | -0.178 |
| **Ours (MSE) − Crossformer** | +0.498 | +0.415 | +0.203 | +0.089 | +0.220 | +0.160 | +0.137 | +0.209 | +0.072 | +0.072 |

### 3c. Share of total squared error contributed by each decile

RMSE is a sum of squares, so a decile matters in proportion to the squared error it carries, not to its width. Share = (sum of squared errors in the decile) / (total), %.

| Variant | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours (MSE) | 7.7% | 6.9% | 6.8% | 6.9% | 7.6% | 7.6% | 8.8% | 9.3% | 12.7% | 25.6% |
| Ours (Huber, headline) | 8.1% | 7.3% | 6.7% | 6.7% | 7.4% | 7.7% | 8.5% | 9.0% | 12.5% | 26.1% |
| Crossformer | 7.0% | 6.4% | 6.7% | 7.1% | 7.5% | 7.6% | 8.9% | 9.2% | 13.1% | 26.5% |

### 3d. Temperature only, by its own amplitude decile

| Variant | metric | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours (MSE) | MAE | 2.438±0.117 | 2.028±0.086 | 1.922±0.074 | 1.917±0.069 | 1.964±0.060 | 1.956±0.053 | 1.966±0.057 | 1.966±0.063 | 2.007±0.044 | 2.244±0.108 |
| Ours (MSE) | RMSE | 3.101±0.164 | 2.560±0.107 | 2.456±0.082 | 2.472±0.080 | 2.553±0.094 | 2.556±0.075 | 2.562±0.060 | 2.589±0.044 | 2.653±0.065 | 2.875±0.145 |
| Ours (Huber, headline) | MAE | 2.358±0.134 | 2.008±0.055 | 1.908±0.073 | 1.916±0.074 | 1.944±0.060 | 1.947±0.069 | 1.939±0.062 | 1.942±0.088 | 1.997±0.065 | 2.228±0.066 |
| Ours (Huber, headline) | RMSE | 3.023±0.170 | 2.550±0.071 | 2.457±0.059 | 2.468±0.069 | 2.542±0.066 | 2.539±0.074 | 2.534±0.053 | 2.559±0.068 | 2.633±0.064 | 2.870±0.095 |
| Crossformer | MAE | 3.839±0.267 | 3.243±0.231 | 3.102±0.212 | 3.037±0.244 | 3.041±0.234 | 2.872±0.212 | 2.827±0.210 | 2.764±0.192 | 2.836±0.209 | 2.878±0.204 |
| Crossformer | RMSE | 4.594±0.267 | 4.055±0.278 | 3.931±0.248 | 3.861±0.283 | 3.864±0.249 | 3.629±0.238 | 3.598±0.239 | 3.508±0.219 | 3.571±0.210 | 3.613±0.225 |

## 4. Tail of the error distribution

Absolute-error quantiles over all (window, horizon, channel) triples, and the share of total squared error carried by the worst 1 % of them.

| Variant | p50 | p90 | p99 | p99.9 | max | SE share of worst 1% |
|---|---|---|---|---|---|---|
| Ours (MSE) | 1.657±0.092 | 11.478±0.334 | 35.039±0.780 | 56.142±1.288 | 79.230±3.716 | 29.813±0.646 |
| Ours (Huber, headline) | 1.565±0.036 | 11.130±0.206 | 36.039±0.294 | 56.859±1.905 | 79.655±3.168 | 30.763±0.739 |
| Crossformer | 1.795±0.076 | 10.713±0.331 | 34.975±1.305 | 55.282±2.403 | 73.585±2.481 | 30.734±1.430 |

## 5. Where the aggregate gap actually comes from: the channel budget

The headline MAE/RMSE are unweighted means over the four targets in physical units. RH has ~10x the error scale of T, so the aggregate is an RH statistic wearing a coat. Aggregate MSE = mean of the four channel MSEs, so each channel moves it by (its MSE gap)/4.

| Variant | MSE T | MSE RH | MSE P | MSE WS | aggregate MSE | aggregate RMSE |
|---|---|---|---|---|---|---|
| Ours (MSE) | 7.002 | 249.340 | 10.858 | 0.898 | 67.025 | 8.187 |
| Ours (Huber, headline) | 6.888 | 252.419 | 9.808 | 0.895 | 67.503 | 8.216 |
| Crossformer | 14.742 | 230.738 | 9.872 | 0.885 | 64.059 | 8.004 |
| **Ours (MSE) − Crossformer** | -7.740 | +18.603 | +0.986 | +0.013 | +2.966 | — |
| **contribution to aggregate MSE gap** | -1.935 | +4.651 | +0.247 | +0.003 | +2.966 | — |

### 5b. RH only, by RH amplitude decile (RMSE)

| Variant | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours (MSE) | 13.817±0.495 | 13.200±0.381 | 13.076±0.352 | 13.201±0.490 | 13.881±0.383 | 13.776±0.438 | 14.849±0.571 | 15.199±0.497 | 17.809±0.368 | 25.096±0.730 |
| Ours (Huber, headline) | 14.366±0.539 | 13.678±0.331 | 13.048±0.250 | 13.060±0.143 | 13.759±0.193 | 13.895±0.147 | 14.673±0.169 | 15.007±0.161 | 17.716±0.187 | 25.538±0.490 |
| Crossformer | 12.428±0.995 | 11.985±0.902 | 12.312±0.573 | 12.729±0.558 | 13.133±0.368 | 13.224±0.258 | 14.382±0.372 | 14.575±0.485 | 17.493±0.629 | 24.881±1.101 |
| **Ours (MSE) − Crossformer** | +1.388 | +1.215 | +0.763 | +0.472 | +0.748 | +0.552 | +0.467 | +0.624 | +0.316 | +0.215 |

## 6. Is the loss the culprit? (Huber operating regime)

Training uses `HuberLoss(delta=1.0)` on globally standardised targets (train sd: T=10.96, RH=25.78, P=10.02, WS=1.21). Fraction of test residuals past delta, i.e. the share the loss treats as L1 rather than L2:

| Variant | overall | T | RH | P | WS |
|---|---|---|---|---|---|
| Ours (MSE) | 0.068 | 0.001 | 0.104 | 0.013 | 0.155 |
| Ours (Huber, headline) | 0.066 | 0.001 | 0.108 | 0.012 | 0.143 |
| Crossformer | 0.062 | 0.004 | 0.095 | 0.013 | 0.137 |

About 93 % of residuals sit in the quadratic region, so the objective is effectively MSE for every model here. The loss does not explain an MAE-good / RMSE-bad profile.

## 7. What does explain it: RH dispersion

Ratio of predicted sd to ground-truth sd per channel, and the scalar `a` that would minimise RH MSE under `pred -> mean + a*(pred - mean)`. `a < 1` means the forecast is over-dispersed for a squared loss.

| Variant | sd ratio T | sd ratio RH | sd ratio P | sd ratio WS | oracle a (RH) |
|---|---|---|---|---|---|
| Ours (MSE) | 1.042 | 0.919 | 1.055 | 0.641 | 0.838 |
| Ours (Huber, headline) | 1.017 | 0.933 | 1.007 | 0.631 | 0.819 |
| Crossformer | 1.099 | 0.849 | 0.947 | 0.546 | 0.916 |

Effect of applying that RH rescale (oracle, fitted **on test** — a measurement of how much of the gap is dispersion, **not** a proposed method; an honest version must fit `a` on validation):

| Variant | aggregate RMSE as-is | with RH rescaled |
|---|---|---|
| Ours (MSE) | 8.185±0.174 | 7.989±0.156 |
| Ours (Huber, headline) | 8.216±0.049 | 7.967±0.046 |
| Crossformer | 8.002±0.172 | 7.953±0.155 |
