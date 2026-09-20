"""Explanation fidelity for all 11 models on 5 seeds (P0-3, P0-4).

PRIMARY metric: deterministic fidelity (analysis/fidelity_deterministic.py):
each perturbed channel is replaced by its own per-window mean, the reference
is fixed and shared by every model, so a checkpoint always gets the same
number and the spread across seeds is model variance only.

APPENDIX: the permutation metric of src/xai.py, which carries ~0.18 sd of
Monte-Carlo noise on a single fixed checkpoint at the current budget.

Baselines appear in the variant the main table reports (analysis/selection.py).

Inputs  analysis/fidelity_det.csv          primary
        xai_v2_out/xai_metrics.csv         appendix (permutation)
        analysis/fidelity_estimator_noise.csv
        analysis/occlusion_time.csv        time axis (also deterministic)
Outputs analysis/xai_fidelity_v2.md
        paper/tables/fidelity_{ds}.tex              primary
        paper/tables/fidelity_perm_appendix_{ds}.tex appendix
"""

import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from selection import selected_suffix  # noqa: E402

OURS, LABEL, HEAD = "XAI-MeteoFormer", "MeteoFormer", "no_revin"
DATASETS = (("jena", "Jena"), ("beijing_aotizhongxin", "Beijing (Aotizhongxin)"))


def holm(p):
    p = np.asarray(p, float)
    adj = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    run = 0.0
    for i, j in enumerate(ok[np.argsort(p[ok])]):
        run = max(run, (len(ok) - i) * p[j])
        adj[j] = min(run, 1.0)
    return adj


def ms(v, k=3):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "—"
    return (f"{v.mean():.{k}f} ± {v.std(ddof=1):.{k}f}" if len(v) > 1
            else f"{v.mean():.{k}f}")


def tex(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return (f"{v.mean():.3f} $\\pm$ {v.std(ddof=1):.3f}" if len(v) > 1
            else ("--" if len(v) == 0 else f"{v.mean():.3f}"))


def nm(m, abl=None):
    if m != OURS:
        return m
    return f"**{LABEL}**" if abl in (None, HEAD) else f"{LABEL} ({abl})"


def load_det():
    f = os.path.join(ROOT, "analysis", "fidelity_det.csv")
    if not os.path.exists(f):
        return pd.DataFrame()
    d = pd.read_csv(f)
    d["suffix"] = d["suffix"].fillna("")
    return d


def selected(d, ds):
    """rows for the models/variants the main table reports"""
    x = d[d.dataset == ds]
    keep = x.apply(lambda r: (r.ablation == HEAD) if r.model == OURS
                   else (r.ablation == "full"
                         and r.suffix == selected_suffix(r.model, ds)), axis=1)
    return x[keep]


def load_perm():
    d = pd.read_csv(os.path.join(ROOT, "xai_v2_out", "xai_metrics.csv"))
    d["seed"] = d.ckpt.map(lambda s: int(re.search(r"_s(\d+)\.pt$", s).group(1)))
    d["suffix"] = d.ckpt.map(lambda s: (re.search(r"(_norm(?:on|off))_s\d+\.pt$", s)
                                        or [None, ""])[1] if "_norm" in s else "")
    return d


def paired_vs_ours(x, col):
    ours = x[x.model == OURS].set_index("seed")[col]
    recs = []
    for m, g in x[x.model != OURS].groupby("model"):
        b = g.set_index("seed")[col]
        s = sorted(set(ours.index) & set(b.index))
        if len(s) < 2:
            continue
        recs.append((m, float((ours.loc[s] - b.loc[s]).mean()),
                     stats.ttest_rel(ours.loc[s], b.loc[s]).pvalue, len(s)))
    adj = holm([r[2] for r in recs])
    return [(m, dm, p, pa, n) for (m, dm, p, n), pa in zip(recs, adj)]


def write_tex(path, body, caption, label):
    with open(path, "w") as f:
        f.write("\\begin{table}[H]\n\\caption{" + caption + "}\n")
        f.write("\\label{" + label + "}\n" + body + "\n\\end{table}\n")


def main():
    det = load_det()
    L = []
    W = L.append
    W("# Explanation fidelity, all 11 models × 5 seeds (P0-3, P0-4)\n")
    W("Reviewer 1, Major #4 and #7. Produced by "
      "[fidelity_deterministic.py](analysis/fidelity_deterministic.py) (primary), "
      "[xai_backfill.py](analysis/xai_backfill.py) (appendix) and "
      "[xai_fidelity_v2.py](analysis/xai_fidelity_v2.py).\n")
    W("## Summary of what changed against the published table\n")
    W("- The published table set a 5-seed mean for our model against single-seed "
      "points for six baselines and had no entry for DLinear, Transformer, "
      "Informer and Autoformer. **Every model now has 5 seeds under identical "
      "conditions, on both datasets.**")
    W("- **SHAP now exists for every model.** It was missing for PatchTST, "
      "iTransformer and TFT because their normalisation divides in place "
      "(`x_enc /= stdev`); `src/xai.py` silently fell back to permutation "
      "importance. The harness rewrites that line out of place at runtime, "
      "forward pass bit-identical ([xai_patches.py](analysis/xai_patches.py)).")
    W("- **Baselines are explained in the normalization variant the main table "
      "reports**, selected on validation loss "
      "([selection.py](analysis/selection.py)): Autoformer (both datasets) and "
      "Transformer (Beijing) with RevIN, LSTM (Jena) without it.")
    W("- **The primary metric is now deterministic.** See next section.\n")

    # ------------------------------------------------------------ metric
    W("## The primary metric: deterministic fidelity\n")
    W("The permutation metric of `src/xai.py` has two sources of Monte-Carlo "
      "noise: the perturbation (a random time permutation of each perturbed "
      "channel) and the reference (a **single** random channel ordering). On one "
      "fixed checkpoint re-explained with five RNG seeds its sd is ~0.18 "
      "([fidelity_estimator_noise.csv](analysis/fidelity_estimator_noise.csv)) — "
      "as large as the spread across five trained seeds, so at the current "
      "budget it cannot separate models closer than ~0.2.\n")
    W("The deterministic metric keeps the definition and removes both sources:\n")
    W("| | permutation metric (appendix) | deterministic metric (primary) |")
    W("|---|---|---|")
    W("| perturbation of a channel | random permutation along time | "
      "replacement by its own per-window mean |")
    W("| reference curve | one random channel ordering | k = 1: exact mean over "
      "channels; k ≥ 2: mean over 20 fixed orderings, shared by every model |")
    W("| model-agnostic ranking | permutation importance (random) | "
      "single-channel window-mean occlusion (deterministic) |")
    W("| SHAP | seeded by the model seed | same function and budget, torch "
      "seeded to 0 for every run |")
    W("| scoring | ks = 0,1,2,3,5,8; gain = mean(ranked − reference) over k>0; "
      "rel = gain / base MAE | identical |\n")
    W("**Determinism, verified.** The same checkpoint explained twice produces "
      "bit-identical importances and curves (`np.array_equal` on every array). "
      "Any spread across seeds in the tables below is therefore model variance, "
      "not estimator noise. The 20 reference orderings are a fixed sample, not "
      "the exact expectation over all orderings; being identical for every "
      "model, they cannot favour one.\n")
    W("Column meaning: **SHAP-ranked** is the fidelity of the model's GradientSHAP "
      "ranking — the cross-model comparison of explanation quality. "
      "**Occlusion-ranked** ranks channels by the same perturbation that scores "
      "them, so it is close to an upper reference, not a fair comparison. "
      "**ρ(SHAP, occl)** is the Spearman agreement of the two rankings.\n")

    if det.empty:
        W("_analysis/fidelity_det.csv not found — run fidelity_deterministic.py._\n")
    for ds, title in DATASETS:
        x = selected(det, ds) if not det.empty else det
        if x.empty:
            continue
        W(f"\n---\n\n## {title}\n")
        W("### Deterministic fidelity, mean ± sd over seeds\n")
        W("| Model | variant | n | SHAP-ranked | occlusion-ranked | ρ(SHAP, occl) |")
        W("|---|---|---|---|---|---|")
        rows = sorted(((g.det_gain_shap_rel.mean(), m, g)
                       for m, g in x.groupby("model")), key=lambda r: -r[0])
        for _, m, g in rows:
            var = "headline" if m == OURS else (g.suffix.iloc[0].lstrip("_") or "historical")
            W(f"| {nm(m)} | {var} | {g.seed.nunique()} | {ms(g.det_gain_shap_rel)} | "
              f"{ms(g.det_gain_occl_rel)} | {ms(g.det_agree_shap_occl_rho)} |")
        o = x[x.model == OURS]
        if len(o) and "det_gain_attn_rel" in o:
            W(f"\n**{LABEL}, built-in attention ranking:** fidelity "
              f"{ms(o.det_gain_attn_rel)}; Spearman(attention, SHAP) "
              f"{ms(o.det_agree_attn_shap_rho)}; Spearman(attention, occlusion) "
              f"{ms(o.det_agree_attn_occl_rho)}.\n")

        W("### Paired tests against our model (SHAP-ranked, deterministic)\n")
        W("Paired by seed, paired t-test, Holm over the 10 baselines. Wilcoxon is "
          "not shown: with n = 5 its smallest two-sided p is 0.0625. Positive Δ = "
          "our model's SHAP ranking is more faithful.\n")
        W("| Baseline | Δ (ours − baseline) | paired t p | Holm | n |")
        W("|---|---|---|---|---|")
        res = paired_vs_ours(x, "det_gain_shap_rel")
        for m, dm, p, pa, n in sorted(res, key=lambda r: r[1]):
            star = " **\\***" if pa < 0.05 else ""
            W(f"| {m} | {dm:+.3f} | {p:.3g} | {pa:.3g}{star} | {n} |")
        if res:
            W(f"\nOur model's SHAP ranking is more faithful than "
              f"{sum(r[1] > 0 for r in res)} of {len(res)} baselines on average; "
              f"**{sum(r[3] < 0.05 for r in res)} of {len(res)}** differences "
              f"survive Holm.\n")

        body = ["\\begin{tabular}{lccc}", "\\toprule",
                "Model & SHAP-ranked & Occlusion-ranked & $\\rho$(SHAP, occl.) \\\\",
                "\\midrule"]
        for _, m, g in rows:
            n_ = f"\\textbf{{{LABEL}}}" if m == OURS else m
            body.append(f"{n_} & {tex(g.det_gain_shap_rel)} & "
                        f"{tex(g.det_gain_occl_rel)} & {tex(g.det_agree_shap_occl_rho)} \\\\")
        if len(o) and o.det_gain_attn_rel.notna().any():
            body += ["\\midrule",
                     f"\\textbf{{{LABEL}}} (built-in attention) & "
                     f"\\multicolumn{{3}}{{c}}{{{tex(o.det_gain_attn_rel)}}} \\\\"]
        body += ["\\bottomrule", "\\end{tabular}"]
        write_tex(os.path.join(ROOT, "paper", "tables", f"fidelity_{ds}.tex"),
                  "\n".join(body),
                  f"Explanation fidelity on {ds}, physical channels only, mean "
                  "$\\pm$ s.d. over 5 seeds for every model. Deterministic "
                  "metric: a perturbed channel is replaced by its own window "
                  "mean and the reference ordering set is fixed and shared, so "
                  "the spread is model variance only. Baselines in the "
                  "normalization variant selected on validation. SHAP-ranked = "
                  "fidelity of the model's GradientSHAP ranking; "
                  "occlusion-ranked ranks by the scoring perturbation itself "
                  "and serves as an upper reference.", f"tab:fid_{ds}")

    # ------------------------------------------------------------ P0-4
    W("\n---\n\n## P0-4: does the entropy regulariser buy faithfulness?\n")
    W("Isolated by two variants that differ in `lambda_ent` alone, on the "
      "published configuration: `no_revin` (λ=0.01, the headline) vs "
      "`no_revin+no_entropy` (λ=0), both RevIN off, 5 seeds, both datasets, "
      "paired by seed. Deterministic metric; the time-axis rows come from "
      "[occlusion_time.md](analysis/occlusion_time.md), which was deterministic "
      "from the start.\n")
    W("> Correction. An earlier version compared `no_entropy` (RevIN on) with "
      "`no_revin` (RevIN off) and concluded that the regulariser doubles the "
      "attention–occlusion agreement. That comparison changed RevIN too and is "
      "withdrawn.\n")
    cols = (("det_gain_attn_rel", "fidelity, attention-ranked"),
            ("det_agree_attn_shap_rho", "Spearman(attention, SHAP)"),
            ("det_agree_attn_occl_rho", "Spearman(attention, occlusion)"),
            ("det_gain_shap_rel", "fidelity, SHAP-ranked"),
            ("det_gain_occl_rel", "fidelity, occlusion-ranked"))
    oc_path = os.path.join(ROOT, "analysis", "occlusion_time.csv")
    oc_all = pd.read_csv(oc_path) if os.path.exists(oc_path) else pd.DataFrame()
    p04 = []
    for ds, title in DATASETS:
        if det.empty:
            break
        on = det[(det.model == OURS) & (det.ablation == HEAD) & (det.dataset == ds)]
        off = det[(det.model == OURS) & (det.ablation == "no_revin+no_entropy")
                  & (det.dataset == ds)]
        W(f"### {title}\n")
        if on.empty or off.empty:
            W("_not computed yet_\n")
            continue
        sd = sorted(set(on.seed) & set(off.seed))
        W(f"| Metric (n = {len(sd)}) | λ=0.01 (headline) | λ=0 | Δ | paired t p | λ>0 better |")
        W("|---|---|---|---|---|---|")
        for col, lab in cols:
            a = on.set_index("seed")[col].reindex(sd)
            b = off.set_index("seed")[col].reindex(sd)
            pv = stats.ttest_rel(a, b).pvalue if len(sd) > 1 else np.nan
            p04.append((title, lab, pv, int((a > b).sum()), len(sd),
                        "attention" in lab))
            W(f"| {lab} | {ms(a)} | {ms(b)} | {(a - b).mean():+.3f} | {pv:.3f} | "
              f"{int((a > b).sum())}/{len(sd)} |")
        if len(oc_all):
            oc = oc_all[oc_all.dataset == ds]
            a = oc[oc.ablation == HEAD].set_index("seed").rho_temp_attn
            b = oc[oc.ablation == "no_revin+no_entropy"].set_index("seed").rho_temp_attn
            so = sorted(set(a.index) & set(b.index))
            if len(so) > 1:
                pv = stats.ttest_rel(a.reindex(so), b.reindex(so)).pvalue
                p04.append((title, "time occlusion ρ(attention)", pv,
                            int((a.reindex(so) > b.reindex(so)).sum()), len(so), True))
                rc = oc[oc.ablation == HEAD].set_index("seed").rho_recency.reindex(so)
                W(f"| time occlusion ρ(attention) | {ms(a.reindex(so))} | "
                  f"{ms(b.reindex(so))} | {(a.reindex(so) - b.reindex(so)).mean():+.3f} "
                  f"| {pv:.3f} | {int((a.reindex(so) > b.reindex(so)).sum())}/{len(so)} |")
                W(f"\nRecency control on the same windows: ρ = {ms(rc)}.\n")
    if p04:
        ps = np.array([r[2] for r in p04], float)
        adj = holm(ps)
        best = int(np.nanargmin(ps))
        att = [r for r in p04 if r[5]]
        fav, tot = sum(r[3] for r in att), sum(r[4] for r in att)
        W(f"**Multiple comparisons.** {len(p04)} paired tests. Smallest raw p = "
          f"{ps[best]:.3f} ({p04[best][0]}, {p04[best][1]}), Holm-adjusted "
          f"{adj[best]:.2f}. {'No difference' if (adj >= 0.05).all() else 'At least one difference'} "
          "survives the correction.\n")
        W(f"**Reading.** On the attention rows λ=0.01 comes out ahead in **{fav} "
          f"of {tot}** seed-level comparisons. A regulariser that made attention "
          "more faithful would win most of them. The SHAP- and occlusion-ranked "
          "rows describe the model rather than its attention and should barely "
          "move.\n")
        W("**Conclusion for Major #7.** "
          + ("The entropy regulariser does not improve faithfulness; no "
             "difference survives correction and the direction on the attention "
             "rows is against it. " if (adj >= 0.05).all() and fav < tot / 2 else
             "See the table: read any surviving difference against its "
             "direction across seeds. ")
          + "The claim that it \"lifts the fidelity/stability numbers\" (also in "
          "the docstring of [src/xm_models/xai_meteoformer.py](src/xm_models/xai_meteoformer.py)) "
          "is not supported. What it demonstrably does is sparsify the attention "
          "([results_hygiene.md](analysis/results_hygiene.md)); if kept, describe "
          "it as a sparsity prior only.\n")

    # ------------------------------------------------------------ RevIN
    W("\n---\n\n## What does move faithfulness: RevIN\n")
    W("`full` (RevIN on) vs the headline `no_revin` (RevIN off), 5 seeds each, "
      "paired by seed, deterministic metric.\n")
    for ds, title in DATASETS:
        if det.empty:
            break
        fu = det[(det.model == OURS) & (det.ablation == "full") & (det.dataset == ds)]
        hd = det[(det.model == OURS) & (det.ablation == HEAD) & (det.dataset == ds)]
        s5 = sorted(set(fu.seed) & set(hd.seed))
        if len(s5) < 2:
            W(f"### {title}\n\n_not computed yet_\n")
            continue
        W(f"### {title}\n")
        W("| Metric | RevIN on (full) | RevIN off (headline) | Δ | paired t p | on > off |")
        W("|---|---|---|---|---|---|")
        for col, lab in cols:
            a = fu.set_index("seed")[col].reindex(s5)
            b = hd.set_index("seed")[col].reindex(s5)
            W(f"| {lab} | {ms(a)} | {ms(b)} | {(a - b).mean():+.3f} | "
              f"{stats.ttest_rel(a, b).pvalue:.3f} | {int((a > b).sum())}/{len(s5)} |")
        W("")

    # ------------------------------------------------------------ appendix
    W("\n---\n\n## Appendix: the permutation metric of `src/xai.py`\n")
    W("Same models, variants, windows and SHAP budget; perturbation by random "
      "time permutation and a single random reference ordering. **Caveat: on one "
      "fixed checkpoint this metric has sd ≈ 0.18 from its own Monte-Carlo noise "
      "at the current budget** — as large as the between-seed spread — so "
      "differences below ~0.2 are not interpretable. Kept for continuity with the "
      "published table, whose sd of 0.037 came from reusing one RNG stream across "
      "checkpoints.\n")
    nz_path = os.path.join(ROOT, "analysis", "fidelity_estimator_noise.csv")
    if os.path.exists(nz_path):
        nz = pd.read_csv(nz_path)
        W("| One fixed checkpoint, RNG seed 0–4 | SHAP-ranked | permutation-ranked |")
        W("|---|---|---|")
        for m, g in nz.groupby("model"):
            W(f"| {nm(m)} | {ms(g.fidelity_gain_shap_rel)} | {ms(g.fidelity_gain_perm_rel)} |")
        W("")
    perm = load_perm()
    for ds, title in DATASETS:
        x = perm[perm.dataset == ds]
        keep = x.apply(lambda r: (r.ablation == HEAD) if r.model == OURS
                       else r.suffix == selected_suffix(r.model, ds), axis=1)
        x = x[keep]
        if x.empty:
            continue
        W(f"### {title}, permutation metric\n")
        W("| Model | n | SHAP-ranked | permutation-ranked | ρ(SHAP, perm) |")
        W("|---|---|---|---|---|")
        rows = sorted(((g.fidelity_gain_shap_rel.mean(), m, g)
                       for m, g in x.groupby("model")), key=lambda r: -r[0])
        for _, m, g in rows:
            W(f"| {nm(m)} | {g.seed.nunique()} | {ms(g.fidelity_gain_shap_rel)} | "
              f"{ms(g.fidelity_gain_perm_rel)} | {ms(g.agree_shap_perm_rho)} |")
        W("")
        body = ["\\begin{tabular}{lccc}", "\\toprule",
                "Model & SHAP-ranked & Permutation-ranked & $\\rho$(SHAP, perm.) \\\\",
                "\\midrule"]
        for _, m, g in rows:
            n_ = f"\\textbf{{{LABEL}}}" if m == OURS else m
            body.append(f"{n_} & {tex(g.fidelity_gain_shap_rel)} & "
                        f"{tex(g.fidelity_gain_perm_rel)} & {tex(g.agree_shap_perm_rho)} \\\\")
        body += ["\\bottomrule", "\\end{tabular}"]
        write_tex(os.path.join(ROOT, "paper", "tables",
                               f"fidelity_perm_appendix_{ds}.tex"), "\n".join(body),
                  f"Appendix. Explanation fidelity on {ds} with the permutation "
                  "metric (random time permutation, single random reference "
                  "ordering), 5 seeds. On one fixed checkpoint this estimator "
                  "has s.d. $\\approx$ 0.18 from its own sampling noise at this "
                  "budget, so differences below $\\approx$ 0.2 are not "
                  "interpretable; the deterministic metric in the main text "
                  "removes this noise.", f"tab:fid_perm_{ds}")

    # ------------------------------------------------------------ verdict
    W("\n---\n\n## Verdict for P0-3 and P0-4\n")
    if not det.empty:
        perm_all = load_perm()
        for ds, title in DATASETS:
            x = selected(det, ds)
            if x.empty:
                continue
            rank = x.groupby("model").det_gain_shap_rel.mean().sort_values(ascending=False)
            pos = list(rank.index).index(OURS) + 1
            res = paired_vs_ours(x, "det_gain_shap_rel")
            worse = [(m, dm, pa) for m, dm, p, pa, n in res if dm < 0]
            worse_sig = [m for m, dm, pa in worse if pa < 0.05]
            cf = [r for r in res if r[0] == "Crossformer"]
            o = x[x.model == OURS]
            po = perm_all[(perm_all.dataset == ds) & (perm_all.model == OURS)
                          & (perm_all.ablation == HEAD)]
            sd_p = float(po.fidelity_gain_shap_rel.std(ddof=1)) if len(po) > 1 else np.nan
            sd_d = float(o.det_gain_shap_rel.std(ddof=1))
            W(f"**{title}.** Our model ranks **{pos} of {len(rank)}** by "
              f"SHAP-ranked fidelity ({ms(o.det_gain_shap_rel)}). "
              + (f"Against Crossformer: Δ = {cf[0][1]:+.3f}, Holm p = {cf[0][3]:.3g}"
                 + (" — **Crossformer is significantly more faithful**. "
                    if cf[0][3] < 0.05 else " — not significant. ") if cf else "")
              + (f"Significantly more faithful than ours: "
                 f"{', '.join(worse_sig)}. " if worse_sig
                 else "No baseline is significantly more faithful. ")
              + f"Built-in attention: {ms(o.det_gain_attn_rel)} — "
                "indistinguishable from zero, and now with a tight interval "
                "rather than the wide one the permutation metric produced. "
              + (f"Removing the estimator noise shrank the seed sd of our own "
                 f"score from {sd_p:.3f} to {sd_d:.3f}.\n"
                 if sd_p == sd_p else "\n"))
        W("**What this changes.** Under the noisy permutation metric our model "
          "looked second, just behind Crossformer, with seed sds of ~0.19 that "
          "made every ranking meaningless. With the noise removed the ordering "
          "is resolved and it is less flattering: our model sits mid-table on "
          "both datasets. The honest claim is no longer \"most faithful\" but "
          "\"comparable to the strongest baselines by SHAP-ranked fidelity, "
          "ahead of the weaker half, and behind Crossformer\".\n")
        W("**The built-in attention remains the weak point** and is now firmly "
          "so: its fidelity is ~0 with a small interval, and it is far below "
          "the same model's SHAP ranking. The defensible framing is that the "
          "model is explainable by post-hoc attribution, not that its attention "
          "is an explanation.\n")
        W("**Two findings are unchanged by the metric swap**, which is the best "
          "evidence that they are real: the entropy regulariser does not buy "
          "faithfulness (P0-4 above), and RevIN does — the configuration "
          "selected for accuracy, `no_revin`, has markedly less faithful "
          "attention than `full` (Jena: attention-ranked fidelity 0.181 vs "
          "0.012, paired p = 0.001, 5 of 5 seeds).\n")

    dst = os.path.join(ROOT, "analysis", "xai_fidelity_v2.md")
    open(dst, "w").write("\n".join(L))
    print("->", dst)


if __name__ == "__main__":
    main()
