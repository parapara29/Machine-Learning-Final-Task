#!/usr/bin/env bash
# Closed-loop evaluation of the V6 DAgger policy on the linear-regression test protocol.
#   1. trained_model_dagger_v6 on all 1,000 cases       -> eval_v6/linear_protocol/v6
#   2. DAgger round checkpoints r0, r1 on cases 0-199  -> eval_v6/linear_protocol/r0, r1
#   3. 4 example episodes with saved trajectories      -> eval_v6/examples
#   4. single-process inference benchmark              -> eval_v6/inference_benchmark.json
#   5. figures + metrics                                -> figures/v6_*.pdf, figures/v6_results.json
# Resumable (finished episodes are skipped). Run from robotnik_dagger_v5/:
#   bash run_v6_evaluation.sh            (or: setsid nohup bash run_v6_evaluation.sh > eval_v6.log 2>&1 &)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
WORKERS=${WORKERS:-10}
CASES=eval_v6/linear_protocol_cases.csv
[ -f "$CASES" ] || "$PY" make_linear_protocol_cases.py --out "$CASES"
COMMON="--mode deploy --cases-csv $CASES --stop-rule linear --max-steps 40000 --workers $WORKERS --resume"

echo "=== V6 on the 1,000 linear-protocol cases ==="
"$PY" robotnik_dagger_collect.py $COMMON --model-dir trained_model_dagger_v6 \
    --start-episode 0 --num-episodes 1000 --output-dir eval_v6/linear_protocol/v6
for r in r0 r1; do
  if [ -d "trained_model_dagger_v6_$r" ]; then
    echo "=== DAgger round checkpoint $r on cases 0-199 ==="
    "$PY" robotnik_dagger_collect.py $COMMON --model-dir "trained_model_dagger_v6_$r" \
        --start-episode 0 --num-episodes 200 --output-dir "eval_v6/linear_protocol/$r"
  fi
done
echo "=== example trajectories ==="
"$PY" robotnik_dagger_collect.py $COMMON --model-dir trained_model_dagger_v6 --save-trajectory --log-every 10 \
    --episode-ids 1 2 5 7 --output-dir eval_v6/examples
echo "=== inference benchmark ==="
"$PY" benchmark_inference.py
echo "=== figures ==="
"$PY" plot_v6_results.py
echo "EVALUATION DONE ($(date))"
