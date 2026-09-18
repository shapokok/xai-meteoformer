#!/bin/sh
# Reproduces analysis/fidelity_estimator_noise.csv: ONE checkpoint per model
# (seed 0), explained five times with RNG seeds 0..4 and nothing else changed.
# The spread is the Monte-Carlo noise of the fidelity estimator itself.
cd "$(dirname "$0")/.." || exit 1
W="import sys, runpy; sys.path.insert(0, 'analysis'); import xai_patches; sys.argv = ['src/xai.py'] + sys.argv[1:]; runpy.run_path('src/xai.py', run_name='__main__')"
OUT=${OUT:-/tmp/fidelity_noise}
for M in XAI-MeteoFormer:no_revin Crossformer:full; do
  mdl=${M%%:*}; abl=${M#*:}
  extra=""; [ "$mdl" != XAI-MeteoFormer ] && extra=--skip_internal
  for r in 0 1 2 3 4; do
    ./.venv/bin/python -c "$W" --dataset jena --model "$mdl" \
      --ckpt "checkpoints/${mdl}_jena_${abl}_s0.pt" --ablation "$abl" \
      --seed "$r" --max_batches 40 --batch_size 64 --exclude_time \
      --tslib_path Time-Series-Library --device mps \
      --out_dir "$OUT/${mdl}_r$r" $extra
  done
done
