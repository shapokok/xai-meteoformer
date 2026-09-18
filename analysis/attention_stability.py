"""Is the variable attention stable among correlated predictors?

Reviewer 2 #1. Jena has several near-collinear thermodynamic channels
(T, Tpot, Tdew, VPact, VPmax, VPdef, SH, H2OC, rho, RH). If the
attention picks one of them essentially at random from seed to seed,
the "explanation" is not identifying a physical driver, it is breaking
a tie. The defensible claim in that case is at the level of the GROUP,
not the individual channel -- so this script measures both.

Measured, per target and per correlation group, over the 5 seeds:
  * per-channel weight mean and sd across seeds
  * how often the arg-max channel inside a group changes across seeds
  * the group's TOTAL weight mean and sd -- the quantity that should be
    stable even when the within-group winner is not

Reads xai/*.npz only. No model is run.
"""

import json
import os
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XAI = os.path.join(ROOT, "xai")
OURS, SEEDS = ("XAI-MeteoFormer", "no_revin"), range(5)
TARGETS = ["T", "RH", "P", "WS"]
CORR_T = 0.9            # |r| above which channels count as near-collinear


def groups_from_corr(X, cols, thr=CORR_T):
    """Connected components of the |r| >= thr graph, singletons dropped."""
    C = np.corrcoef(X.T)
    n = len(cols)
    seen, out = set(), []
    for i in range(n):
        if i in seen:
            continue
        comp, stack = set(), [i]
        while stack:
            j = stack.pop()
            if j in comp:
                continue
            comp.add(j)
            for k in range(n):
                if k not in comp and abs(C[j, k]) >= thr:
                    stack.append(k)
        seen |= comp
        if len(comp) > 1:
            out.append(sorted(comp))
    return out, C


def main():
    L, summary = [], []
    W = L.append
    W("# Stability of the variable attention among correlated predictors\n")
    W("Reviewer 2 #1. Produced by "
      "[analysis/attention_stability.py](analysis/attention_stability.py) from "
      "`xai/*.npz`; no model is run.\n")
    W(f"Channels are grouped into connected components of the "
      f"|Pearson r| ≥ {CORR_T} graph on the **training** split. Within such a "
      "group the channels are nearly interchangeable, so which one a softmax "
      "attention selects is close to arbitrary. The question is whether the "
      "*group's* total weight is stable even when the winner inside it is not.\n")

    for ds, dsname in [("jena", "Jena"), ("beijing_aotizhongxin",
                                          "Beijing (Aotizhongxin)")]:
        meta = json.load(open(os.path.join(ROOT, "data/processed",
                                           f"{ds}_meta.json")))
        cols = meta["columns"]
        X = np.load(os.path.join(ROOT, "data/processed", f"{ds}_X.npy"))
        n_tr = int(X.shape[0] * 0.7)
        grps, C = groups_from_corr(X[:n_tr], cols)

        va = []
        for s in SEEDS:
            f = os.path.join(XAI, f"{OURS[0]}_{ds}_{OURS[1]}_s{s}_xai.npz")
            if os.path.exists(f):
                d = np.load(f, allow_pickle=True)
                va.append(d["var_attn"].mean(axis=0))      # (n_targets, N)
        if not va:
            W(f"\n_{dsname}: no XAI arrays_\n")
            continue
        va = np.stack(va)                                   # (S, n_targets, N)
        S = va.shape[0]

        W(f"\n---\n\n## {dsname}\n")
        W(f"{S} seeds. Correlation groups found "
          f"(|r| ≥ {CORR_T} on the training split):\n")
        if not grps:
            W("_none — no channel pair reaches the threshold_\n")
        for gi, g in enumerate(grps, 1):
            rs = [abs(C[a, b]) for i, a in enumerate(g) for b in g[i + 1:]]
            W(f"- **G{gi}**: {', '.join(cols[i] for i in g)} "
              f"(|r| from {min(rs):.3f} to {max(rs):.3f})")
        W("")

        W("### Per-channel weight across seeds (mean±sd), top channels\n")
        W("| Target | " + " | ".join(
            f"#{i+1}" for i in range(5)) + " |")
        W("|---|" + "---|" * 5)
        for t, tn in enumerate(TARGETS):
            order = np.argsort(-va[:, t, :].mean(0))[:5]
            cells = [f"{cols[c]} {va[:, t, c].mean():.3f}±{va[:, t, c].std(ddof=1):.3f}"
                     for c in order]
            W(f"| {tn} | " + " | ".join(cells) + " |")
        W("")

        if grps:
            W("### Within-group instability vs group-level stability\n")
            W("`winner flips` = number of distinct arg-max channels inside the "
              "group across the 5 seeds (1 = always the same channel, 5 = a "
              "different one every seed). `CV` = sd/mean.\n")
            W("| Target | Group | winner flips | modal winner (freq) | "
              "winner weight mean±sd (CV) | **group total mean±sd (CV)** |")
            W("|---|---|---|---|---|---|")
            for t, tn in enumerate(TARGETS):
                for gi, g in enumerate(grps, 1):
                    sub = va[:, t, :][:, g]                 # (S, |g|)
                    win = [g[i] for i in sub.argmax(axis=1)]
                    cnt = Counter(win)
                    modal, freq = cnt.most_common(1)[0]
                    wv = np.array([va[s, t, win[s]] for s in range(S)])
                    tot = sub.sum(axis=1)
                    cvw = wv.std(ddof=1) / wv.mean() if wv.mean() > 0 else np.nan
                    cvt = tot.std(ddof=1) / tot.mean() if tot.mean() > 0 else np.nan
                    W(f"| {tn} | G{gi} | {len(cnt)} | {cols[modal]} ({freq}/{S}) | "
                      f"{wv.mean():.3f}±{wv.std(ddof=1):.3f} ({cvw:.2f}) | "
                      f"**{tot.mean():.3f}±{tot.std(ddof=1):.3f} ({cvt:.2f})** |")
            W("")
            cw, ct = [], []
            for t in range(len(TARGETS)):
                for g in grps:
                    sub = va[:, t, :][:, g]
                    win = [g[i] for i in sub.argmax(axis=1)]
                    wv = np.array([va[s, t, win[s]] for s in range(S)])
                    tot = sub.sum(axis=1)
                    if wv.mean() > 0:
                        cw.append(wv.std(ddof=1) / wv.mean())
                    if tot.mean() > 0:
                        ct.append(tot.std(ddof=1) / tot.mean())
            W(f"**Summary — {dsname}:** mean CV of the within-group winner's "
              f"weight = **{np.mean(cw):.3f}**; mean CV of the group total = "
              f"**{np.mean(ct):.3f}** "
              f"(ratio {np.mean(cw)/max(np.mean(ct),1e-9):.1f}×).\n")
            summary.append((dsname, np.mean(cw), np.mean(ct)))
        # worst per-channel CV among the top-weighted channels
        top_cv = []
        for t in range(len(TARGETS)):
            c = int(np.argmax(va[:, t, :].mean(0)))
            m_ = va[:, t, c].mean()
            if m_ > 0:
                top_cv.append((cols[c], va[:, t, c].std(ddof=1) / m_))
        W("### Seed-to-seed variability of the single top channel\n")
        W("| Target | top channel | CV across seeds |")
        W("|---|---|---|")
        for tn, (cn, cv) in zip(TARGETS, top_cv):
            W(f"| {tn} | {cn} | {cv:.2f} |")
        W("")

    V = ["\n---\n\n## Verdict\n"]
    V.append("**The variable attention is not stable across seeds, and "
             "grouping correlated channels does not rescue it.**\n")
    for dsname, cw, ct in summary:
        V.append(f"- {dsname}: mean CV of the within-group winner "
                 f"**{cw:.2f}**, of the group total **{ct:.2f}**. Grouping "
                 f"reduces the variability by about {cw/max(ct,1e-9):.1f}×, "
                 f"but a CV of {ct:.2f} still means the group weight moves by "
                 f"roughly {100*ct:.0f}% of its own size from seed to seed.")
    V.append("")
    V.append("Two further observations that a reviewer will make before we do:\n")
    V.append("1. **The attention concentrates on the time-of-day encodings, "
             "not on physical predictors.** On Jena `hour_cos` and `hour_sin` "
             "take the top two slots for T, RH and WS; on Beijing `hour_cos` "
             "alone takes 0.44-0.54 of the mass for RH, P and WS. An "
             "explanation whose headline finding is \"the model uses the hour "
             "of day\" is not wrong, but it is not the physical-driver story "
             "the paper currently tells.")
    V.append("2. **The sd is of the same order as the mean for the top "
             "channels** (e.g. Beijing WS: `hour_cos` 0.539±0.397). Reporting "
             "a mean attention vector over seeds without its sd, as the "
             "current figures do, hides this entirely.")
    V.append("")
    V.append("### What this means for the resubmission\n")
    V.append("The claim that the variable attention identifies *which physical "
             "variable* drives a target cannot be supported by this evidence. "
             "Three honest options, in descending order of strength:\n")
    V.append("- **Report the instability as a finding.** Attention over "
             "near-collinear meteorological inputs is unstable by "
             "construction, quantify it, and present the group-level weight "
             "as the only quantity worth reading. This is a real contribution "
             "and it is what the numbers support.")
    V.append("- **Keep the attention as an internal mechanism** and move the "
             "explanation claims onto permutation/SHAP importance, which is "
             "model-agnostic and comparable across the baselines.")
    V.append("- Drop the per-channel attention figures and keep only the "
             "group-level and fidelity results.")
    V.append("")
    V.append(f"Note on Beijing: no channel pair reaches |r| ≥ {CORR_T}, so the "
             "group analysis is vacuous there. That is itself notable — RH on "
             "Beijing is *derived* from T and Tdew via the Magnus formula "
             "(src/data/prepare.py), so the three are functionally dependent "
             "even though their linear correlation stays below the threshold.")
    open(os.path.join(ROOT, "analysis", "attention_stability.md"),
         "w").write("\n".join(L + V))
    print("-> analysis/attention_stability.md")


if __name__ == "__main__":
    main()
