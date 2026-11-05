#!/usr/bin/env bash
# Evaluate the V5.4 checkpoint against V5.3-A.
#  1. V5.4 on the 300 validation targets (V5.3-A baseline: eval_v53_validation_fresh/).
#  2. Both models on the same 100 random targets + random spawns (5 shards x 20, seeds 1000..1004).
# Resumable: every shard writes its own output dir and a finished shard (deploy_summary.csv with
# the expected row count) is skipped on re-run. Run from robotnik_dagger_v5/:
#   setsid nohup bash run_v54_evaluation.sh > eval_v54.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/rick/LAB_SESSION_COMP_0245_2026_PUBLIC/RoboEnv/.pixi/envs/ml/bin/python}
JOBS=${JOBS:-6}

jobs_file=$(mktemp)
# Each line: <expected_rows> <output_dir> <collector args...>
# Validation split -> 30 shards of 10 episodes
"$PY" - <<'PY' >> "$jobs_file"
import json
ids = json.load(open('trained_model_dagger_v5_3_A/episode_split.json'))['validation']
for k in range(0, len(ids), 10):
    shard = ids[k:k+10]
    out = f"eval_v54_validation/shard_{k//10:02d}"
    print(f"{len(shard)} {out} --model-dir trained_model_dagger_v5_4 "
          f"--targets-csv ../data_randomized/targets.csv --episode-ids {' '.join(map(str, shard))}")
PY
# Random target + spawn, identical seeds for both models
for model in trained_model_dagger_v5_3_A trained_model_dagger_v5_4; do
  for k in 0 1 2 3 4; do
    echo "20 eval_random/${model#trained_model_dagger_}/shard_$k --model-dir $model --random-target --random-spawn --start-episode $((k*20)) --num-episodes 20 --seed $((1000+k))" >> "$jobs_file"
  done
done

run_shard() {
  local expected=$1 out=$2; shift 2
  local summary="$out/deploy_summary.csv"
  if [ -f "$summary" ] && [ $(( $(wc -l < "$summary") - 1 )) -ge "$expected" ]; then
    echo "skip (done): $out"; return 0
  fi
  rm -rf "$out"   # partial shard: start it again from scratch
  "$PY" robotnik_dagger_collect.py --mode deploy --output-dir "$out" "$@" > /dev/null 2>&1 \
    && echo "done: $out" || echo "FAILED: $out"
}
export -f run_shard; export PY

echo "Running $(wc -l < "$jobs_file") shards with $JOBS workers ($(date))"
xargs -P "$JOBS" -L 1 bash -c 'run_shard "$@"' _ < "$jobs_file"
rm -f "$jobs_file"
echo "ALL EVALUATION DONE ($(date))"
