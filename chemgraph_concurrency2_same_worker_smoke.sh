#!/usr/bin/env bash
set -euo pipefail

# Launch two concurrent ChemGraph smoke suites on the same worker node while
# sharing one vLLM server on a separate node.

ROOT="${CHEMGRAPH_LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORKER_NODE="${CHEMGRAPH_WORKER_NODE:-x3208c0s7b1n0}"
VLLM_NODE="${VLLM_NODE:-x3208c0s7b0n0}"
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL="${CHEMGRAPH_MODEL:-chemgraph-qwen3-32b}"
SMOKE_SCRIPT="${CHEMGRAPH_SMOKE_SCRIPT:-$ROOT/chemgraph_separate_node_vllm_exp_smoke.sh}"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_ROOT="${CHEMGRAPH_CONCURRENCY_LOG_ROOT:-$ROOT/ChemGraph/cg_logs/concurrency2_same_worker_$STAMP}"
COMBINED_SUMMARY="$RUN_ROOT/combined_summary.csv"

[[ -x "$SMOKE_SCRIPT" ]] || { echo "Missing executable smoke script: $SMOKE_SCRIPT" >&2; exit 2; }
mkdir -p "$RUN_ROOT"

echo "RUN_ROOT=$RUN_ROOT"
echo "WORKER_NODE=$WORKER_NODE"
echo "VLLM_NODE=$VLLM_NODE"
echo "VLLM_PORT=$VLLM_PORT"
echo "MODEL=$MODEL"
echo "SMOKE_SCRIPT=$SMOKE_SCRIPT"

ssh -o BatchMode=yes -o ConnectTimeout=8 "$WORKER_NODE" \
  "hostname; curl --noproxy '*' -fsS http://$VLLM_NODE:$VLLM_PORT/v1/models >/dev/null"

pids=()
workers=()

launch_worker() {
  local worker_id="$1"
  local logdir="$RUN_ROOT/$worker_id"
  local launcher_log="$RUN_ROOT/$worker_id.launcher.log"

  mkdir -p "$logdir"
  echo "Launching $worker_id on $WORKER_NODE; logdir=$logdir"

  ssh -o BatchMode=yes -o ConnectTimeout=8 "$WORKER_NODE" \
    "CHEMGRAPH_LOCAL_ROOT='$ROOT' VLLM_NODE='$VLLM_NODE' VLLM_PORT='$VLLM_PORT' CHEMGRAPH_MODEL='$MODEL' CHEMGRAPH_SMOKE_LOG_DIR='$logdir' '$SMOKE_SCRIPT'" \
    >"$launcher_log" 2>&1 &

  pids+=("$!")
  workers+=("$worker_id")
}

launch_worker worker_1
launch_worker worker_2

overall_status=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  worker_id="${workers[$i]}"
  if wait "$pid"; then
    echo "$worker_id process finished"
  else
    echo "$worker_id process failed" >&2
    overall_status=1
  fi
done

printf 'worker,id,status,seconds,workflow,recursion_limit,log\n' >"$COMBINED_SUMMARY"
for worker_id in "${workers[@]}"; do
  summary="$RUN_ROOT/$worker_id/summary.csv"
  if [[ ! -f "$summary" ]]; then
    echo "Missing summary for $worker_id: $summary" >&2
    overall_status=1
    continue
  fi

  tail -n +2 "$summary" | while IFS= read -r line; do
    printf '%s,%s\n' "$worker_id" "$line"
  done >>"$COMBINED_SUMMARY"
done

echo
echo "Combined summary: $COMBINED_SUMMARY"
cat "$COMBINED_SUMMARY"

if grep -q ',FAIL,' "$COMBINED_SUMMARY" || grep -q ',TIMEOUT,' "$COMBINED_SUMMARY"; then
  overall_status=1
fi

if [[ "$overall_status" -eq 0 ]]; then
  echo "Concurrency smoke result: PASS"
else
  echo "Concurrency smoke result: FAIL" >&2
fi

exit "$overall_status"
