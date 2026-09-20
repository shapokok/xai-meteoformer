"""Render analysis/occlusion_time.md from analysis/occlusion_time.csv."""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = 11


def block(d, L):
    W = L.append
    sp = [d[f"span_p{i}"].iloc[0] for i in range(P)]
    dm = np.array([[r[f"dmae_p{i}"] for i in range(P)] for _, r in d.iterrows()])
    at = np.array([[r[f"attn_p{i}"] for i in range(P)] for _, r in d.iterrows()])
    ro = np.array([[r[f"roll_p{i}"] for i in range(P)] for _, r in d.iterrows()])

    W("| patch | hours | occlusion ΔMAE | temporal attention | rollout |")
    W("|---|---|---|---|---|")
    for i in range(P):
        W(f"| p{i} | {sp[i]} | **{dm[:, i].mean():+.4f}** ± {dm[:, i].std(ddof=1):.4f} "
          f"| {at[:, i].mean():.4f} ± {at[:, i].std(ddof=1):.4f} "
          f"| {ro[:, i].mean():.4f} ± {ro[:, i].std(ddof=1):.4f} |")
    W("")
    am, aM = at.mean(0).min(), at.mean(0).max()
    W(f"Attention spread: **{am:.4f} … {aM:.4f}**, i.e. "
      f"{100*(aM-am)/at.mean():.1f}% of its own mean, against a uniform "
      f"value of 1/{P} = {1/P:.4f}. Occlusion importance spans "
      f"{dm.mean(0).min():+.4f} … {dm.mean(0).max():+.4f}.\n")
    return dm, at, ro


def main():
    d0 = pd.read_csv(os.path.join(ROOT, "analysis", "occlusion_time.csv"))
    L = []
    W = L.append
    W("# Occlusion: external validation of the temporal axis\n")
    W("Reviewer 1, Major #3/#4/#7. Produced by "
      "[analysis/occlusion_time.py](analysis/occlusion_time.py) and "
      "[analysis/occlusion_report.py](analysis/occlusion_report.py). "
      "Inference only, on existing checkpoints.\n")
    W("**Method.** For each of the 11 input patch positions (patch_len=16, "
      "stride=8), the patch is replaced in every channel by the per-channel "
      "mean of the rest of that window, the model is re-run, and the rise in "
      "test MAE is recorded. Replacing by the window's own mean keeps the "
      "level and destroys only the local detail; zeroing would inject a "
      "spurious level shift in scaled space. 40 test batches × 64 = 2560 "
      "windows, the same subsample `src/xai.py` uses.\n")
    W("This is the external ground truth the paper is missing: it is measured "
      "from the model's *behaviour*, not read out of the same attention "
      "computation it is supposed to validate.\n")
    W("**The recency control.** A trivial ordering — later patches matter more "
      "— is included as a baseline explanation. Any attention mechanism that "
      "does not beat it is contributing nothing.\n")

    W("**Which rows to compare.** The entropy regulariser is isolated only by "
      "variants that differ in `lambda_ent` alone. Primary control, on the "
      "published configuration: `no_revin` vs `no_revin+no_entropy` (both RevIN "
      "off). Secondary: `full` vs `no_entropy` (both RevIN on). Setting "
      "`no_entropy` against the headline changes RevIN too — an earlier reading "
      "of this file did exactly that and is withdrawn; see "
      "[xai_fidelity_v2.md](analysis/xai_fidelity_v2.md).\n")
    variants = [("no_revin", "Headline model (no_revin: RevIN off, entropy on)"),
                ("no_revin+no_entropy", "no_revin+no_entropy (RevIN off, entropy off) — "
                                        "clean control for the regulariser"),
                ("full", "full (RevIN on, entropy on)"),
                ("no_entropy", "no_entropy (RevIN on, entropy off)")]
    for ds, dtitle in (("jena", "Jena"), ("beijing_aotizhongxin", "Beijing")):
      for abl, name in variants:
        d = d0[(d0.dataset == ds) & (d0.ablation == abl)]
        if d.empty:
            continue
        W(f"\n---\n\n## {name} — {dtitle}, {len(d)} seeds\n")
        W("### Spearman agreement with the occlusion ranking\n")
        W("| | mean ρ | sd | per seed |")
        W("|---|---|---|---|")
        for c, lab in [("rho_temp_attn", "temporal attention"),
                       ("rho_rollout", "attention rollout"),
                       ("rho_recency", "**recency control**")]:
            W(f"| {lab} | {d[c].mean():+.3f} | {d[c].std(ddof=1):.3f} | "
              + ", ".join(f"{v:+.3f}" for v in d[c]) + " |")
        W("")
        if len(d) > 2:
            for c, lab in [("rho_temp_attn", "attention"),
                           ("rho_rollout", "rollout")]:
                diff = (d[c] - d["rho_recency"])
                p = stats.ttest_rel(d[c], d["rho_recency"]).pvalue
                W(f"- paired {lab} − recency: **{diff.mean():+.4f}** "
                  f"(t-test p = {p:.3f})")
            W("")
        W("### Where the model actually looks\n")
        block(d, L)

    d = d0[(d0.dataset == "jena") & (d0.ablation == "no_revin")]
    if not d.empty:
        dm = np.array([[r[f"dmae_p{i}"] for i in range(P)] for _, r in d.iterrows()])
        last = dm.mean(0)[-1]
        rest = np.abs(dm.mean(0)[:-1]).sum()
        W("\n---\n\n## Verdict\n")
        W("**The temporal attention does not survive external validation.**\n")
        W(f"1. **The model is a last-16-hours model.** Occluding the final "
          f"patch (t = 80–96 h) costs **{last:+.3f} MAE**. Occluding all ten "
          f"earlier patches costs {rest:.3f} in total absolute terms, and "
          f"several of them are *negative* — removing them slightly **improves** "
          f"the forecast, i.e. the model treats that part of the window as "
          f"noise. The effective receptive field is one patch wide.")
        W(f"2. **The attention is flat and therefore uninformative.** It spans "
          f"0.0905–0.0924 across the eleven patches — a 2% spread around the "
          f"uniform value 1/11 — while true importance spans three orders of "
          f"magnitude. It assigns essentially the same weight to the patch "
          f"worth +2.04 MAE and to patches worth 0.00.")
        W(f"3. **Rank correlation flatters it badly.** ρ(attention) = "
          f"{d.rho_temp_attn.mean():+.3f} looks respectable, but it is high "
          f"only because the minute monotone drift in the attention happens to "
          f"order the patches correctly. Spearman is blind to magnitude, and "
          f"the magnitudes are where the explanation fails.")
        W(f"4. **It does not beat the recency control.** ρ(attention) = "
          f"{d.rho_temp_attn.mean():+.3f} vs ρ(recency) = "
          f"{d.rho_recency.mean():+.3f}, paired difference "
          f"{(d.rho_temp_attn - d.rho_recency).mean():+.4f}, p = "
          f"{stats.ttest_rel(d.rho_temp_attn, d.rho_recency).pvalue:.2f}. "
          f"Rollout is *worse* than recency "
          f"({d.rho_rollout.mean():+.3f}).")
        W(f"5. **It is unstable across seeds**, ρ ranging "
          f"{d.rho_temp_attn.min():.3f}–{d.rho_temp_attn.max():.3f}, which "
          f"matches the variable-attention instability in "
          f"[analysis/attention_stability.md](analysis/attention_stability.md).")
        W("")
        W("### What to do\n")
        W("- Do not claim the temporal attention explains *when* the model "
          "looks. The occlusion profile says the honest finding is: the model "
          "uses the last 16 hours and effectively ignores the rest of the "
          "96-hour window.")
        W("- That finding is publishable and useful — it is a concrete, "
          "externally validated statement about the model, and it raises an "
          "obvious question for the window-length sweep (item 6 of the "
          "addendum): if only the last 16 h matter, `seq_len`=96 is mostly "
          "wasted context, and `seq_len`=24 or 48 should lose little.")
        W("- Report the occlusion curve as the temporal-importance figure and "
          "keep the attention as an internal mechanism, with its low fidelity "
          "stated. This mirrors the recommendation for the variable axis, where "
          "permutation/SHAP fidelity is 0.52–0.54 against 0.18 for the built-in "
          "attention.")
        W("- The recency control belongs in the paper. Any reviewer will ask "
          "for it once occlusion is shown.")

    dst = os.path.join(ROOT, "analysis", "occlusion_time.md")
    open(dst, "w").write("\n".join(L))
    print("->", dst)


if __name__ == "__main__":
    main()
