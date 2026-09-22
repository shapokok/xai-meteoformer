"""Figures 2 and 4 of the manuscript, redrawn for the headline configuration.

Both figures in the submitted version were produced from `full` (RevIN on) and
carried the internal model name inside the image. The paper's headline model is
the `no_revin` ablation, labelled "MeteoFormer", so both are redrawn from the
artefacts of that configuration.

  Figure 2  one test window of Jena (index 4000), all four targets, observed
            against MeteoFormer / Crossformer / DLinear, each the mean over its
            5 seeds. Baselines in the normalization variant the main table
            reports (analysis/selection.py).
  Figure 4  channel importance on Jena for the 15 physical channels: the
            model's own variable attention, GradientSHAP, and single-channel
            occlusion (replace the channel by its own window mean), mean ± sd
            over 5 seeds. Each method is normalised to a share of its own total
            so three different units can share one axis.

Outputs -> paper/figures/fig2_forecast_window_jena.png   (1600x1000)
        -> paper/figures/fig4_channel_importance_jena.png (1600x1000)
"""

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from selection import selected_ablation  # noqa: E402

PRED = os.path.join(ROOT, "predictions")
XAI = os.path.join(ROOT, "xai")
XAI_DET = os.path.join(ROOT, "xai_det")
FIG = os.path.join(ROOT, "paper", "figures")
DS = "jena"
WINDOW = 4000
SEEDS = range(5)
OURS = "XAI-MeteoFormer"

# categorical slots 1-3 of the validated palette; observed is text ink, not a
# category, so it gets the neutral colour
INK, MUTED = "#1a1a19", "#6b6a63"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TARGETS = [("T", "Temperature (°C)"), ("RH", "Relative humidity (%)"),
           ("P", "Pressure (mbar)"), ("WS", "Wind speed (m s$^{-1}$)")]

plt.rcParams.update({
    "font.size": 8, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 200,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def seed_mean_pred(model):
    abl = selected_ablation(model, DS)
    ps = [np.load(os.path.join(PRED, f"{model}_{DS}_{abl}_s{s}_pred.npy"))
          for s in SEEDS]
    return np.mean(ps, axis=0)


def figure_2():
    true = np.load(os.path.join(PRED, f"{DS}_true.npy"))
    series = [("MeteoFormer", seed_mean_pred(OURS), BLUE),
              ("Crossformer", seed_mean_pred("Crossformer"), ORANGE),
              ("DLinear", seed_mean_pred("DLinear"), AQUA)]
    h = np.arange(1, true.shape[1] + 1)

    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    for ax, (code, label) in zip(axes.ravel(), TARGETS):
        c = [t for t, _ in TARGETS].index(code)
        ax.plot(h, true[WINDOW, :, c], color=INK, lw=2.0, label="Observed",
                zorder=5)
        for name, p, colour in series:
            ax.plot(h, p[WINDOW, :, c], color=colour, lw=1.6, label=name,
                    zorder=4)
        ax.set_title(label, fontsize=8, color=INK, loc="left")
        ax.set_xlim(1, h[-1])
        ax.set_xticks([1, 6, 12, 18, 24])
    for ax in axes[1]:
        ax.set_xlabel("Forecast horizon (h)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Jena, test window {WINDOW}: 24-hour forecast, mean of 5 seeds",
                 fontsize=9, color=INK, y=1.0)
    fig.tight_layout()
    p = os.path.join(FIG, "fig2_forecast_window_jena.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def importance_vectors():
    """per-seed, per-channel importance of the three methods, each normalised
    to a share of its own total over the 15 physical channels"""
    z0 = np.load(os.path.join(XAI, f"{OURS}_{DS}_no_revin_s0_xai.npz"),
                 allow_pickle=True)
    names = [str(c) for c in z0["channels"]]
    out = {"attention": [], "SHAP": [], "occlusion": []}
    for s in SEEDS:
        z = np.load(os.path.join(XAI, f"{OURS}_{DS}_no_revin_s{s}_xai.npz"),
                    allow_pickle=True)
        d = np.load(os.path.join(XAI_DET, f"{OURS}_{DS}_no_revin_s{s}_det.npz"),
                    allow_pickle=True)
        occl = np.asarray(d["occl"], dtype=float)
        n = len(occl)                                  # physical channels only
        for key, vec in (("attention", np.asarray(z["attn_importance"], float)[:n]),
                         ("SHAP", np.asarray(z["shap_importance"], float)[:n]),
                         ("occlusion", occl)):
            v = np.abs(vec)
            out[key].append(v / v.sum())
    return names[:len(out["attention"][0])], {k: np.array(v) for k, v in out.items()}


def figure_4():
    names, imp = importance_vectors()
    order = np.argsort(imp["SHAP"].mean(axis=0))          # ascending, barh
    y = np.arange(len(order))
    methods = [("Variable attention (built-in)", "attention", BLUE),
               ("GradientSHAP", "SHAP", ORANGE),
               ("Occlusion (window mean)", "occlusion", AQUA)]
    height = 0.26

    fig, ax = plt.subplots(figsize=(8, 5))
    for k, (label, key, colour) in enumerate(methods):
        m = imp[key].mean(axis=0)[order]
        sd = imp[key].std(axis=0, ddof=1)[order]
        ax.barh(y + (k - 1) * height, m, height=height * 0.92, color=colour,
                label=label, zorder=3)
        ax.errorbar(m, y + (k - 1) * height, xerr=sd, fmt="none", ecolor=MUTED,
                    elinewidth=0.7, capsize=1.5, zorder=4)
    # a share cannot be negative; the whisker on `rho` runs past zero because
    # its sd exceeds its mean (attention: 0.211 +/- 0.299), which is the
    # instability reported in analysis/attention_stability.md. Clip the axis
    # and name the case instead of letting one error bar set the scale.
    wide = [(i, imp["attention"].mean(axis=0)[j], imp["attention"].std(axis=0, ddof=1)[j])
            for i, j in enumerate(order)
            if imp["attention"].std(axis=0, ddof=1)[j] > imp["attention"].mean(axis=0)[j]]
    for i, m_, sd_ in wide:
        ax.annotate(f"{m_:.2f} ± {sd_:.2f}", (m_ + sd_ * 0.06, i - height),
                    fontsize=6.5, color=MUTED, va="center")
    ax.set_xlim(0, None)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("Share of the method's total importance over the 15 physical channels\n"
                  "(error bars: ±1 s.d. over seeds; where the s.d. exceeds the mean the "
                  "value is printed)")
    ax.set_title("Jena, MeteoFormer (headline, no RevIN): channel importance, "
                 "mean ± s.d. over 5 seeds", fontsize=9, loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(-0.6, len(order) - 0.4)
    fig.tight_layout()
    p = os.path.join(FIG, "fig4_channel_importance_jena.png")
    fig.savefig(p)
    plt.close(fig)
    return p


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    for p in (figure_2(), figure_4()):
        from PIL import Image
        print("->", p, Image.open(p).size)
