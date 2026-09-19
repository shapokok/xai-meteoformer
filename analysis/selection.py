"""Which normalization variant of each baseline the paper reports.

The rule, fixed in advance in analysis/norm_symmetry_runs.md and adopted
for the main table: per (model, dataset), the variant with the lower mean
VALIDATION loss over the 5 seeds -- the same rule that selected no_revin
for our model. Baselines without a second variant keep the historical one.
The historical configuration (LSTM/GRU with our RevIN, every other baseline
as its source defines it) goes to the appendix.

Every analysis that loads a baseline checkpoint or prediction by tag must
go through selected_suffix() / selected_norm(), so the main table and every
downstream analysis report the same model.
"""

import functools
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROPOSED = "XAI-MeteoFormer"
LOCAL_RNN = ("LSTM", "GRU")


def historical_norm(model: str) -> str:
    return "on" if model in LOCAL_RNN else "off"


@functools.lru_cache(maxsize=None)
def _table():
    d = pd.read_csv(os.path.join(ROOT, "analysis", "results_clean.csv"))
    d = d[(d.model != PROPOSED) & (d.ablation == "full") & (d.width == "default")]
    g = d.groupby(["model", "dataset", "norm_variant"]).best_val.agg(["mean", "count"])
    return g.reset_index()


def selected_norm(model: str, dataset: str) -> str:
    """'on' or 'off' for a baseline; always 'off' for our model."""
    if model == PROPOSED:
        return "off"
    t = _table()
    t = t[(t.model == model) & (t.dataset == dataset)]
    if t.norm_variant.nunique() < 2:
        return historical_norm(model)
    return str(t.loc[t["mean"].idxmin(), "norm_variant"])


def selected_suffix(model: str, dataset: str) -> str:
    """Checkpoint/prediction tag suffix of the selected variant, as written
    by src/train.py: empty for the historical variant, else _norm{on,off}."""
    if model == PROPOSED:
        return ""
    nv = selected_norm(model, dataset)
    return "" if nv == historical_norm(model) else f"_norm{nv}"


def selected_ablation(model: str, dataset: str) -> str:
    """The `ablation` value an analysis harness writes for the reported
    variant: no_revin for our model, full[_normon|_normoff] for a baseline."""
    if model == PROPOSED:
        return "no_revin"
    return "full" + selected_suffix(model, dataset)


def keep_selected(d: pd.DataFrame) -> pd.DataFrame:
    """Rows of a harness CSV (model, dataset, ablation) that belong to the
    variant the main table reports."""
    want = [selected_ablation(m, ds) for m, ds in zip(d["model"], d["dataset"])]
    return d[d["ablation"].values == pd.Series(want).values]


def selected_tag(model: str, dataset: str, seed: int) -> str:
    abl = "no_revin" if model == PROPOSED else "full"
    return f"{model}_{dataset}_{abl}{selected_suffix(model, dataset)}_s{seed}"


if __name__ == "__main__":
    t = _table()
    for ds in sorted(t.dataset.unique()):
        for m in sorted(t[t.dataset == ds].model.unique()):
            nv = selected_norm(m, ds)
            flag = "" if nv == historical_norm(m) else "   <- switched"
            print(f"{ds:22s} {m:14s} {nv}{flag}")
