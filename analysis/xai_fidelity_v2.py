"""Explanation fidelity for all 11 models on 5 seeds, uniformly (P0-3, P0-4).

Reads xai_v2_out/xai_metrics.csv, produced by analysis/xai_backfill.py
under identical conditions for every model: 40 test batches x 64 = 2560
explained windows, GradientSHAP with 16 samples on 10 batches, the same
fidelity k-grid, physical channels only, permutation RNG seeded by the
model seed. Seed i of every model therefore sees the same windows and the
same permutation stream, which is what makes the per-seed pairing in the
tests below a genuine pairing rather than an index match.

Replaces the published fidelity table, which set a 5-seed mean for the
proposed model against single-seed points for six baselines and had no
entry at all for DLinear, Transformer, Informer and Autoformer.

Outputs -> analysis/xai_fidelity_v2.md, paper/tables/fidelity_{ds}.tex
"""

import os
import re

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OURS, LABEL = "XAI-MeteoFormer", "MeteoFormer"
HEAD = "no_revin"


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(run, 1.0)
    return adj


def ms(v, k=3):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "—"
    return f"{v.mean():.{k}f} ± {v.std(ddof=1):.{k}f}" if len(v) > 1 else f"{v.mean():.{k}f}"


def load():
    d = pd.read_csv(os.path.join(ROOT, "xai_v2_out", "xai_metrics.csv"))
    d["seed"] = d.ckpt.map(lambda s: int(re.search(r"_s(\d+)\.pt$", s).group(1)))
    return d


def main():
    d = load()
    old = pd.read_csv(os.path.join(ROOT, "xai_metrics.csv"))
    L = []
    W = L.append
    W("# Explanation fidelity, all 11 models × 5 seeds (P0-3, P0-4)\n")
    W("Reviewer 1, Major #4 and #7. Produced by "
      "[analysis/xai_backfill.py](analysis/xai_backfill.py) and "
      "[analysis/xai_fidelity_v2.py](analysis/xai_fidelity_v2.py).\n")
    W("## What changed against the published table\n")
    W("The published fidelity table set a **5-seed mean** for our model against "
      "**single-seed points** for six baselines, and had **no entry at all** "
      "for DLinear, Transformer, Informer and Autoformer. Everything is now "
      "recomputed from scratch under identical conditions for all 11 models:\n")
    W("- 5 seeds per model, both datasets — 110 runs, none failed;")
    W("- 2 560 explained test windows (40 batches × 64), the same windows for "
      "every model;")
    W("- GradientSHAP with 16 samples on 10 batches for every model;")
    W("- the same fidelity k-grid, physical channels only;")
    W("- the permutation RNG seeded by the model seed, so seed *i* of every "
      "model sees the same permutation stream — the per-seed pairing in the "
      "tests below is a real pairing, not an index coincidence.\n")
    W("**SHAP now exists for every model.** It was missing for PatchTST, "
      "iTransformer and TFT in the published table because their normalisation "
      "divides in place (`x_enc /= stdev`), which breaks input gradients; "
      "`src/xai.py` caught the error and silently fell back to permutation "
      "importance. The harness rewrites that one line out of place at runtime "
      "(forward pass bit-identical, checked with `torch.equal`); see "
      "[analysis/xai_patches.py](analysis/xai_patches.py). Neither `src/` nor "
      "the pinned TSLib is modified on disk.\n")
    nz = os.path.join(ROOT, "analysis", "fidelity_estimator_noise.csv")
    if os.path.exists(nz):
        nz = pd.read_csv(nz)
        W("## First: the fidelity estimator is dominated by its own noise\n")
        W("To separate model differences from Monte-Carlo noise, **one** "
          "checkpoint per model (seed 0) was explained five times with RNG "
          "seeds 0–4 and nothing else changed "
          "([fidelity_estimator_noise.csv](analysis/fidelity_estimator_noise.csv), "
          "reproducer [fidelity_estimator_noise.sh](analysis/fidelity_estimator_noise.sh)):\n")
        W("| Model (one fixed checkpoint) | SHAP-ranked | permutation-ranked | "
          "attention-ranked |")
        W("|---|---|---|---|")
        for m, g in nz.groupby("model"):
            nm_ = f"**{LABEL}**" if m == OURS else m
            W(f"| {nm_} | {ms(g.fidelity_gain_shap_rel)} | "
              f"{ms(g.fidelity_gain_perm_rel)} | {ms(g.fidelity_gain_attn_rel)} |")
        W("")
        W("The spread on a **single fixed model** (sd ≈ 0.17–0.18) is as large "
          "as the spread across five independently trained seeds in the tables "
          "below (0.18–0.19). Almost all of the seed-to-seed variance is "
          "estimator noise, not model difference. Two consequences:\n")
        W("1. **The published sd of 0.037 for our model was an artefact.** The "
          "published runs evidently used the same RNG stream for every "
          "checkpoint: for seed 0 the old and new permutation rankings agree "
          "exactly (ρ = 1.000), for seed 1 they do not (ρ = 0.816). The same "
          "random draw repeated five times hides the estimator's noise, and the "
          "published mean of 0.523 is one draw of it.")
        W("2. **At the current budget (2 560 windows, one permutation per "
          "batch) the metric cannot separate models whose fidelity differs by "
          "less than roughly 0.2.** Rankings inside that band are noise.\n")
        W("The fix is a lower-variance estimator, applied uniformly: replace "
          "the random permutation of the top-k channels with a deterministic "
          "replacement by each channel's own window mean — the same device "
          "used for the time axis in "
          "[occlusion_time.md](analysis/occlusion_time.md) — which has no "
          "Monte-Carlo noise at all. That is a change of metric definition, so "
          "it is proposed here, not applied.\n")

    W("`fidelity gain (rel)` = relative error increase when the top-ranked "
      "channels are perturbed, beyond what a random ranking produces. Higher = "
      "the ranking identifies channels the model actually depends on.\n")

    for ds, title in (("jena", "Jena"), ("beijing_aotizhongxin",
                                         "Beijing (Aotizhongxin)")):
        x = d[(d.dataset == ds) & ((d.model != OURS) | (d.ablation == HEAD))]
        if x.empty:
            continue
        W(f"\n---\n\n## {title}\n")
        W("### Fidelity by ranking method, mean ± sd over 5 seeds\n")
        W("| Model | n | SHAP-ranked | permutation-ranked | "
          "Spearman(SHAP, perm) |")
        W("|---|---|---|---|---|")
        rows = []
        for m, g in x.groupby("model"):
            rows.append((g.fidelity_gain_shap_rel.mean(), m, g))
        for _, m, g in sorted(rows, key=lambda r: -r[0]):
            nm = f"**{LABEL}**" if m == OURS else m
            W(f"| {nm} | {g.seed.nunique()} | {ms(g.fidelity_gain_shap_rel)} | "
              f"{ms(g.fidelity_gain_perm_rel)} | {ms(g.agree_shap_perm_rho)} |")
        W("")
        o = x[x.model == OURS]
        if "fidelity_gain_attn_rel" in o and o.fidelity_gain_attn_rel.notna().any():
            W(f"**{LABEL}, built-in attention ranking:** fidelity "
              f"{ms(o.fidelity_gain_attn_rel)}; Spearman(attention, SHAP) "
              f"{ms(o.get('agree_attn_shap_rho', pd.Series(dtype=float)))}; "
              f"Spearman(attention, permutation) "
              f"{ms(o.get('agree_attn_perm_rho', pd.Series(dtype=float)))}.\n")

        W("### Paired tests against our model (SHAP-ranked fidelity)\n")
        W("Paired by seed. Paired t-test, Holm-corrected over the 10 baselines; "
          "Wilcoxon signed-rank alongside. **With n = 5 the smallest two-sided "
          "Wilcoxon p attainable is 0.0625**, so it cannot reach 0.05 whatever "
          "the effect; it is reported for completeness. Positive Δ = our "
          "model's SHAP ranking is more faithful.\n")
        W("| Baseline | Δ (ours − baseline) | paired t p | Holm | Wilcoxon p |")
        W("|---|---|---|---|---|")
        ours = o.set_index("seed").fidelity_gain_shap_rel
        recs = []
        for m, g in x[x.model != OURS].groupby("model"):
            b = g.set_index("seed").fidelity_gain_shap_rel
            s = sorted(set(ours.index) & set(b.index))
            dv = ours.loc[s].values - b.loc[s].values
            tp = stats.ttest_rel(ours.loc[s], b.loc[s]).pvalue
            try:
                wp = stats.wilcoxon(ours.loc[s], b.loc[s]).pvalue
            except ValueError:
                wp = np.nan
            recs.append((m, dv.mean(), tp, wp))
        adj = holm([r[2] for r in recs])
        for (m, dm, tp, wp), pa in sorted(zip(recs, adj), key=lambda z: z[0][1]):
            star = " **\\***" if pa < 0.05 else ""
            W(f"| {m} | {dm:+.3f} | {tp:.3g} | {pa:.3g}{star} | {wp:.3g} |")
        W("")
        n_sig = sum(pa < 0.05 for pa in adj)
        n_pos = sum(r[1] > 0 for r in recs)
        W(f"Our model's SHAP ranking is more faithful than {n_pos} of 10 "
          f"baselines on average; **{n_sig} of 10** differences survive Holm.\n")

        # old vs new, for the channels the paper printed
        oo = old[(old.dataset == ds) & (old.exclude_time == True)]
        if len(oo) and "fidelity_gain_perm_rel" in oo:
            W("### Published table vs recomputed (permutation-ranked)\n")
            W("| Model | published (seeds) | recomputed, 5 seeds |")
            W("|---|---|---|")
            for m, g in x.groupby("model"):
                sub = oo[(oo.model == m) & ((oo.model != OURS) | (oo.ablation == HEAD))]
                pub = (f"{sub.fidelity_gain_perm_rel.mean():.3f} ({len(sub)})"
                       if len(sub) else "— (absent)")
                nm = f"**{LABEL}**" if m == OURS else m
                W(f"| {nm} | {pub} | {ms(g.fidelity_gain_perm_rel)} |")
            W("")

        body = ["\\begin{tabular}{lccc}", "\\toprule",
                "Model & SHAP-ranked & Permutation-ranked & "
                "$\\rho$(SHAP, perm) \\\\", "\\midrule"]
        for _, m, g in sorted(rows, key=lambda r: -r[0]):
            nm = f"\\textbf{{{LABEL}}}" if m == OURS else m
            t = lambda v: (f"{v.mean():.3f} $\\pm$ {v.std(ddof=1):.3f}"
                           if v.notna().sum() > 1 else "--")
            body.append(f"{nm} & {t(g.fidelity_gain_shap_rel)} & "
                        f"{t(g.fidelity_gain_perm_rel)} & "
                        f"{t(g.agree_shap_perm_rho)} \\\\")
        if len(o) and o.fidelity_gain_attn_rel.notna().any():
            v = o.fidelity_gain_attn_rel
            body.append("\\midrule")
            body.append(f"\\textbf{{{LABEL}}} (built-in attention) & "
                        f"\\multicolumn{{3}}{{c}}{{{v.mean():.3f} $\\pm$ "
                        f"{v.std(ddof=1):.3f}}} \\\\")
        body += ["\\bottomrule", "\\end{tabular}"]
        with open(os.path.join(ROOT, "paper", "tables", f"fidelity_{ds}.tex"), "w") as f:
            f.write("\\begin{table}[H]\n\\caption{Explanation fidelity on "
                    f"{ds}, physical channels only, mean $\\pm$ s.d. over 5 "
                    "seeds for every model. All models are explained under "
                    "identical conditions (same 2560 test windows, same "
                    "attribution budget, same perturbation grid). Higher = "
                    "the ranking identifies channels the model depends on.}\n"
                    f"\\label{{tab:fid_{ds}}}\n" + "\n".join(body) +
                    "\n\\end{table}\n")

    # ---------------- P0-4: entropy regulariser, correct control --------- #
    W("\n---\n\n## P0-4: does the entropy regulariser buy faithfulness? (Jena)\n")
    W("**Control.** `no_entropy` is defined relative to `full` "
      "(`ABLATIONS[\"no_entropy\"] = {}` in [src/train.py](src/train.py)), so "
      "it has **RevIN on**. The headline model is `no_revin`. Comparing "
      "`no_entropy` with the headline would change two things at once — the "
      "regulariser *and* RevIN. The only clean comparison is **`full` vs "
      "`no_entropy`**, which differ in `lambda_ent` alone (0.01 vs 0). The "
      "headline is shown for context only.\n")
    W("> Correction. An earlier version of this analysis compared `no_entropy` "
      "with `no_revin` and concluded that the regulariser roughly doubles the "
      "attention–occlusion agreement. That comparison was confounded by RevIN "
      "and is withdrawn; the numbers below replace it.\n")
    fu = d[(d.model == OURS) & (d.ablation == "full") & (d.dataset == "jena")]
    ne = d[(d.model == OURS) & (d.ablation == "no_entropy") & (d.dataset == "jena")]
    hd = d[(d.model == OURS) & (d.ablation == HEAD) & (d.dataset == "jena")]
    if ne.empty or fu.empty:
        W("_full or no_entropy XAI runs not available yet._\n")
    else:
        s_ = sorted(set(ne.seed) & set(fu.seed))
        W(f"`no_entropy` has **{ne.seed.nunique()} seeds**; the clean "
          f"comparison uses the matched seeds {s_}. Seeds 3–4 and all of "
          "Beijing are queued in the GPU track, so this is **provisional**.\n")
        W("| Metric | full (λ=0.01) | no_entropy (λ=0) | Δ full − no_entropy | "
          "headline no_revin (context) |")
        W("|---|---|---|---|---|")
        for col, lab in (("fidelity_gain_attn_rel", "fidelity, attention-ranked"),
                         ("agree_attn_shap_rho", "Spearman(attention, SHAP)"),
                         ("agree_attn_perm_rho", "Spearman(attention, perm)"),
                         ("fidelity_gain_shap_rel", "fidelity, SHAP-ranked"),
                         ("fidelity_gain_perm_rel", "fidelity, permutation-ranked")):
            if col not in d:
                continue
            a = fu.set_index("seed")[col].reindex(s_)
            b = ne.set_index("seed")[col].reindex(s_)
            W(f"| {lab} | {ms(a)} | {ms(b)} | {(a - b).mean():+.3f} | "
              f"{ms(hd[col])} |")
        W("")
        oc = os.path.join(ROOT, "analysis", "occlusion_time.csv")
        if os.path.exists(oc):
            oc = pd.read_csv(oc)
            oc = oc[oc.dataset == "jena"]
            W("Occlusion along time, same code for all three:\n")
            W("| Variant | n | ρ(attention) | ρ(rollout) | ρ(recency control) |")
            W("|---|---|---|---|---|")
            for abl in ("full", "no_entropy", HEAD):
                g = oc[oc.ablation == abl]
                if g.empty:
                    W(f"| {abl} | 0 | _pending_ | | |")
                    continue
                W(f"| {abl} | {len(g)} | {ms(g.rho_temp_attn)} | "
                  f"{ms(g.rho_rollout)} | {ms(g.rho_recency)} |")
            W("")
        W("**Reading, with the noise caveat above in mind.** With n = 3 and an "
          "estimator sd of ~0.18 per run, only a difference well above ~0.3 "
          "could be taken seriously. Treat this table as a direction, not a "
          "result, until the remaining seeds arrive.\n")
    # --------- the effect that IS there: RevIN on vs off, 5 vs 5 --------- #
    if not fu.empty and not hd.empty:
        W("### What does move faithfulness: RevIN, not the regulariser\n")
        W("`full` (RevIN on) and the headline `no_revin` (RevIN off) both have "
          "5 seeds and differ only in RevIN. Paired by seed — identical "
          "explained windows and, within a pair, the same permutation stream, "
          "so much of the estimator noise cancels in the difference (common "
          "random numbers). That is why these paired tests resolve effects "
          "smaller than the ~0.18 unpaired sd would suggest.\n")
        W("| Metric | RevIN on (full) | RevIN off (headline) | Δ | paired t p "
          "| on > off |")
        W("|---|---|---|---|---|---|")
        s5 = sorted(set(fu.seed) & set(hd.seed))
        for col, lab in (("fidelity_gain_attn_rel", "fidelity, attention-ranked"),
                         ("agree_attn_shap_rho", "Spearman(attention, SHAP)"),
                         ("fidelity_gain_shap_rel", "fidelity, SHAP-ranked"),
                         ("fidelity_gain_perm_rel", "fidelity, permutation-ranked")):
            a = fu.set_index("seed")[col].reindex(s5)
            b = hd.set_index("seed")[col].reindex(s5)
            pv = stats.ttest_rel(a, b).pvalue
            W(f"| {lab} | {ms(a)} | {ms(b)} | {(a - b).mean():+.3f} | "
              f"{pv:.3f} | {int((a > b).sum())}/{len(s5)} |")
        W("")
        W("**The configuration selected for accuracy is the one with the least "
          "faithful variable attention.** Switching RevIN off — which the "
          "validation loss chose, and which gives the paper its first place by "
          "MAE — takes attention-ranked fidelity from 0.34 to 0.04 (p = 0.004, "
          "5 of 5 seeds) and the attention–SHAP agreement from 0.84 to 0.48 "
          "(5 of 5 seeds). This is an accuracy–faithfulness trade-off, and it "
          "is the most defensible XAI finding in this analysis: it is paired, "
          "on equal budgets, and consistent across every seed.\n")
        W("On the time axis the picture is different and weaker: neither "
          "configuration's temporal attention beats the recency control in "
          "[occlusion_time.md](analysis/occlusion_time.md), so there is no "
          "faithfulness on that axis to trade.\n")

    W("### Verdict for P0-3 and P0-4\n")
    W("1. The 5-seed-vs-1-seed comparison is gone: every model now has 5 "
      "seeds under identical conditions, and SHAP exists for all 11.")
    W("2. Our model is **not** the most faithful by SHAP or permutation "
      "fidelity: Crossformer is level (Δ −0.02, p = 0.27). It beats 9 of 10 "
      "baselines on average, but only TimesNet and Autoformer survive Holm on "
      "Jena. The published claim of a clear lead rested on a frozen RNG stream.")
    W("3. The built-in attention of the headline model is not faithful "
      "(0.04, indistinguishable from zero).")
    W("4. The entropy regulariser has **no detectable effect** on faithfulness "
      "on either axis in the clean `full` vs `no_entropy` comparison. Its "
      "claimed benefit should be dropped or reworded as sparsity only.")
    W("5. What does matter is RevIN: turning it off bought accuracy and cost "
      "attention faithfulness. That trade-off is the honest XAI story for the "
      "resubmission.\n")

    W("### Structural issue with the ablation table\n")
    W("Every ablation in [src/train.py](src/train.py) is defined relative to "
      "`full`, which has RevIN **on**. The headline model is `no_revin`. So "
      "each ablation row measures what a component contributes to a model the "
      "paper does not report. The clean fix is to re-base the ablations on the "
      "headline configuration (`use_revin=False` plus the component removed). "
      "That is a GPU-track change and would replace P1-2 as currently "
      "specified; it is flagged here, not done.\n")
    W("And, as recorded in [results_hygiene.md](analysis/results_hygiene.md), "
      "`no_var_attn` disables the entropy term too — with `use_var_attn=False` "
      "the attention is a constant and `var_entropy` has no gradient path — so "
      "that row removes two things at once.\n")

    dst = os.path.join(ROOT, "analysis", "xai_fidelity_v2.md")
    open(dst, "w").write("\n".join(L))
    print("->", dst)


if __name__ == "__main__":
    main()
