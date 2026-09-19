# Input-window sweep (P0-2)

Produced by [analysis/window_sweep.py](analysis/window_sweep.py). Headline model (no RevIN), everything else fixed, prediction length 24 h. L = 96 is the published configuration. Mean ± sd over seeds; p (Welch) is Welch's t-test on seed MAE against L = 96. DM is Diebold–Mariano against L = 96 on the test windows common to both lengths (the windows are aligned at the end of the test split; the script asserts the targets match), per-window loss averaged over seeds, Holm over the three lengths within each dataset and loss. Negative Δ favours L.

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

## Reading

Negative ΔMAE means the shorter/longer window is better than the published 96 h. Consistent with [analysis/occlusion_time.md](analysis/occlusion_time.md), which finds the model relies on the last ~16 h of the window.
