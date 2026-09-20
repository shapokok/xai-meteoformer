"""Recompute explanation fidelity for EVERY model on EVERY seed, uniformly.

Reviewer 1, Major #4. The published fidelity table mixes a 5-seed mean
for the proposed model with single-seed points for the baselines, and
four baselines (DLinear, Transformer, Informer, Autoformer) have no XAI
arrays at all. This driver removes both problems by recomputing
everything from scratch under identical conditions:

  * the same number of explained samples   (--max_batches x --batch_size)
  * the same perturbation settings         (fidelity ks, permutation rng)
  * the same attribution budget            (GradientSHAP n_samples)
  * the same channel subset                (--exclude_time: physical only)
  * the same rng seed per run              (= the model seed)

Results go to a NEW directory so the existing xai/ is untouched:
    xai_v2/*.npz, analysis/xai_metrics_v2.csv

Nothing is trained; this runs src/xai.py on the existing checkpoints.
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from selection import selected_suffix  # noqa: E402
OURS = "XAI-MeteoFormer"
# Every model now runs on MPS: analysis/xai_patches.py works around the
# Tensor.unfold bug and the in-place division that blocked GradientSHAP.
CPU_ONLY = set()
# src/xai.py is run through this wrapper so the patches are in place before
# any model is built. src/ itself is not modified.
WRAPPER = ("import sys, runpy; sys.path.insert(0, {analysis!r}); "
           "import xai_patches; "
           "sys.argv = [{xai!r}] + sys.argv[1:]; "
           "runpy.run_path({xai!r}, run_name='__main__')")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=["jena", "beijing_aotizhongxin"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ablations", nargs="*", default=None,
                    help="for the proposed model; default: no_revin (headline)")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--max_batches", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "xai_v2_out"))
    a = ap.parse_args()

    ck = os.path.join(ROOT, "checkpoints")
    npz_dir = os.path.join(a.out_dir, "xai")
    os.makedirs(npz_dir, exist_ok=True)

    jobs = []
    for ds in a.datasets:
        models = a.models or sorted({f.split(f"_{ds}_")[0]
                                     for f in os.listdir(ck)
                                     if f.endswith(".pt") and f"_{ds}_" in f})
        for m in models:
            abls = (a.ablations if (m == OURS and a.ablations)
                    else (["no_revin"] if m == OURS else ["full"]))
            # baselines are explained in the variant the main table reports
            sfx = "" if m == OURS else selected_suffix(m, ds)
            for abl in abls:
                for s in a.seeds:
                    f = os.path.join(ck, f"{m}_{ds}_{abl}{sfx}_s{s}.pt")
                    if os.path.exists(f):
                        jobs.append((ds, m, abl, s, f))

    print(f"{len(jobs)} runs -> {a.out_dir}", flush=True)
    t0 = time.time()
    failed = []
    for i, (ds, m, abl, s, f) in enumerate(jobs, 1):
        tag = os.path.basename(f)[:-3]
        if os.path.exists(os.path.join(npz_dir, f"{tag}_xai.npz")):
            print(f"[{i}/{len(jobs)}] skip {tag}", flush=True)
            continue
        xai = os.path.join(ROOT, "src", "xai.py")
        cmd = [sys.executable, "-c",
               WRAPPER.format(analysis=os.path.join(ROOT, "analysis"), xai=xai),
               "--dataset", ds, "--model", m, "--ckpt", f,
               "--ablation", abl, "--seed", str(s),
               "--max_batches", str(a.max_batches),
               "--batch_size", str(a.batch_size),
               "--exclude_time",
               "--tslib_path", os.path.join(ROOT, "Time-Series-Library"),
               "--processed_dir", os.path.join(ROOT, "data/processed"),
               "--out_dir", a.out_dir,
               "--device", "cpu" if m in CPU_ONLY else "mps"]
        if m != OURS:
            cmd.append("--skip_internal")
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            failed.append(tag)
            print(f"[{i}/{len(jobs)}] FAIL {tag}: "
                  f"{r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '?'}",
                  flush=True)
            continue
        shap_note = ""
        if "GradientSHAP unavailable" in r.stdout:
            shap_note = "  [SHAP UNAVAILABLE -> permutation fallback]"
        print(f"[{i}/{len(jobs)}] {tag:52s} {time.time()-t:6.1f}s"
              f"{shap_note}", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min; failures: {failed}")


if __name__ == "__main__":
    main()
