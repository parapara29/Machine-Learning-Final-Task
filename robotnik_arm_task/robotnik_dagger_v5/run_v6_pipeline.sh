#!/usr/bin/env bash
# V6 DAgger pipeline: wide random spawns, egocentric features, DAgger on wheels AND arm.
#
#   bash run_v6_pipeline.sh            # run every stage (finished stages are skipped)
#   bash run_v6_pipeline.sh train_r0   # run one stage only
#
# Stages (each resumable; progress bars show in the terminal):
#   r0        expert drives (beta=1), wide random spawns          -> arm_dagger_v6_r0
#   train_r0  demonstrations (1400 training episodes) + r0         -> trained_model_dagger_v6_r0
#   r1        wheels+arm mixed (beta=0.5, 0.3 s intervals), learner v6_r0 -> arm_dagger_v6_r1
#   train_r1  + r1 (warm start from r0)                           -> trained_model_dagger_v6_r1
#   r2        wheels+arm learner only (beta=0), learner v6_r1     -> arm_dagger_v6_r2
#   train_r2  + r2 (warm start from r1)                           -> trained_model_dagger_v6 (final)
#   eval      run_v6_evaluation.sh: 1,000 linear-protocol cases + figures/v6_*.pdf
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
WORKERS=${WORKERS:-6}
N_R0=${N_R0:-400}; N_R1=${N_R1:-600}; N_R2=${N_R2:-600}; N_EVAL=${N_EVAL:-200}
WIDE="--random-target --random-spawn --log-every 10 --nav-timeout 45 --workers $WORKERS --resume"

collect() {  # collect <out_dir> <start_id> <n> <seed> <extra args...>
  local out=$1 start=$2 n=$3 seed=$4; shift 4
  echo "=== collect -> $out ($n episodes) ==="
  "$PY" robotnik_dagger_collect.py --mode collect $WIDE --output-dir "$out" \
      --start-episode "$start" --num-episodes "$n" --seed "$seed" "$@"
}

train() {  # train <out_dir> <extra dagger dirs...> [-- --init-from X]
  local out=$1; shift
  if [ -f "$out/v6_metadata.json" ] && [ -f "$out/arm_reach.pth" ]; then
    echo "=== $out already trained, skipping (delete it to retrain) ==="; return; fi
  echo "=== train -> $out ==="
  "$PY" train_v6.py --output-dir "$out" "$@"
}

stage() { [ -z "${ONLY:-}" ] || [ "$ONLY" = "$1" ]; }
ONLY=${1:-}

if stage r0; then
  collect arm_dagger_v6_r0 100000 "$N_R0" 100 --wheel-source expert --arm-source expert --beta 1.0
fi
if stage train_r0; then
  train trained_model_dagger_v6_r0 --dagger-dirs arm_dagger_v6_r0
fi
if stage r1; then
  collect arm_dagger_v6_r1 200000 "$N_R1" 200 --wheel-source mix --arm-source mix \
      --beta 0.5 --mix-interval-steps 300 --model-dir trained_model_dagger_v6_r0
fi
if stage train_r1; then
  train trained_model_dagger_v6_r1 --dagger-dirs arm_dagger_v6_r0 arm_dagger_v6_r1 \
      --init-from trained_model_dagger_v6_r0
fi
if stage r2; then
  collect arm_dagger_v6_r2 300000 "$N_R2" 300 --wheel-source learner --arm-source learner \
      --beta 0.0 --model-dir trained_model_dagger_v6_r1
fi
if stage train_r2; then
  train trained_model_dagger_v6 --dagger-dirs arm_dagger_v6_r0 arm_dagger_v6_r1 arm_dagger_v6_r2 \
      --init-from trained_model_dagger_v6_r1
fi
if stage eval; then
  PY="$PY" WORKERS="$WORKERS" bash run_v6_evaluation.sh
fi
echo "Pipeline finished: ${ONLY:-all stages}"
