# Occlusion: external validation of the temporal axis

Reviewer 1, Major #3/#4/#7. Produced by [analysis/occlusion_time.py](analysis/occlusion_time.py) and [analysis/occlusion_report.py](analysis/occlusion_report.py). Inference only, on existing checkpoints.

**Method.** For each of the 11 input patch positions (patch_len=16, stride=8), the patch is replaced in every channel by the per-channel mean of the rest of that window, the model is re-run, and the rise in test MAE is recorded. Replacing by the window's own mean keeps the level and destroys only the local detail; zeroing would inject a spurious level shift in scaled space. 40 test batches × 64 = 2560 windows, the same subsample `src/xai.py` uses.

This is the external ground truth the paper is missing: it is measured from the model's *behaviour*, not read out of the same attention computation it is supposed to validate.

**The recency control.** A trivial ordering — later patches matter more — is included as a baseline explanation. Any attention mechanism that does not beat it is contributing nothing.


---

## Headline model (no_revin) — Jena, 5 seeds

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

## no_entropy (entropy regulariser off) — Jena, 3 seeds

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

## Verdict

**The temporal attention does not survive external validation.**

1. **The model is a last-16-hours model.** Occluding the final patch (t = 80–96 h) costs **+2.039 MAE**. Occluding all ten earlier patches costs 0.241 in total absolute terms, and several of them are *negative* — removing them slightly **improves** the forecast, i.e. the model treats that part of the window as noise. The effective receptive field is one patch wide.
2. **The attention is flat and therefore uninformative.** It spans 0.0905–0.0924 across the eleven patches — a 2% spread around the uniform value 1/11 — while true importance spans three orders of magnitude. It assigns essentially the same weight to the patch worth +2.04 MAE and to patches worth 0.00.
3. **Rank correlation flatters it badly.** ρ(attention) = +0.578 looks respectable, but it is high only because the minute monotone drift in the attention happens to order the patches correctly. Spearman is blind to magnitude, and the magnitudes are where the explanation fails.
4. **It does not beat the recency control.** ρ(attention) = +0.578 vs ρ(recency) = +0.582, paired difference -0.0036, p = 0.95. Rollout is *worse* than recency (+0.456).
5. **It is unstable across seeds**, ρ ranging 0.355–0.891, which matches the variable-attention instability in [analysis/attention_stability.md](analysis/attention_stability.md).

### What to do

- Do not claim the temporal attention explains *when* the model looks. The occlusion profile says the honest finding is: the model uses the last 16 hours and effectively ignores the rest of the 96-hour window.
- That finding is publishable and useful — it is a concrete, externally validated statement about the model, and it raises an obvious question for the window-length sweep (item 6 of the addendum): if only the last 16 h matter, `seq_len`=96 is mostly wasted context, and `seq_len`=24 or 48 should lose little.
- Report the occlusion curve as the temporal-importance figure and keep the attention as an internal mechanism, with its low fidelity stated. This mirrors the recommendation for the variable axis, where permutation/SHAP fidelity is 0.52–0.54 against 0.18 for the built-in attention.
- The recency control belongs in the paper. Any reviewer will ask for it once occlusion is shown.