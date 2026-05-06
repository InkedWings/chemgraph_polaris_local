#!/usr/bin/env bash
set -euo pipefail

# Profile two concurrent ChemGraph smoke suites on one worker node while using
# a vLLM backend on a separate node.

ROOT="${CHEMGRAPH_LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORKER_NODE="${CHEMGRAPH_WORKER_NODE:-x3208c0s7b1n0}"
VLLM_NODE="${VLLM_NODE:-x3208c0s7b0n0}"
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL="${CHEMGRAPH_MODEL:-chemgraph-qwen3-32b}"
SAMPLE_INTERVAL="${CHEMGRAPH_PROFILE_INTERVAL:-1}"
CONCURRENCY_SCRIPT="${CHEMGRAPH_CONCURRENCY_SCRIPT:-$ROOT/chemgraph_concurrency2_same_worker_smoke.sh}"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_ROOT="${CHEMGRAPH_PROFILE_LOG_ROOT:-$ROOT/ChemGraph/cg_logs/profile_concurrency2_$STAMP}"
SMOKE_ROOT="$RUN_ROOT/smoke"

mkdir -p "$RUN_ROOT"

[[ -x "$CONCURRENCY_SCRIPT" ]] || { echo "Missing executable concurrency script: $CONCURRENCY_SCRIPT" >&2; exit 2; }
[[ -x "$ROOT/chemgraph_profile_cpu_sampler.py" ]] || { echo "Missing CPU sampler" >&2; exit 2; }
[[ -x "$ROOT/chemgraph_profile_gpu_sampler.py" ]] || { echo "Missing GPU sampler" >&2; exit 2; }
[[ -x "$ROOT/chemgraph_profile_report.py" ]] || { echo "Missing profile report script" >&2; exit 2; }

echo "RUN_ROOT=$RUN_ROOT"
echo "SMOKE_ROOT=$SMOKE_ROOT"
echo "WORKER_NODE=$WORKER_NODE"
echo "VLLM_NODE=$VLLM_NODE"
echo "VLLM_PORT=$VLLM_PORT"
echo "MODEL=$MODEL"
echo "SAMPLE_INTERVAL=$SAMPLE_INTERVAL"

ssh -o BatchMode=yes -o ConnectTimeout=8 "$WORKER_NODE" \
  "hostname; curl --noproxy '*' -fsS http://$VLLM_NODE:$VLLM_PORT/v1/models >/dev/null"

VLLM_PID="$(
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$VLLM_NODE" \
    "pgrep -f 'vllm serve.*--port $VLLM_PORT' | head -1" |
  awk '/^[0-9]+$/ { print; exit }'
)"
[[ -n "$VLLM_PID" ]] || { echo "Could not find vLLM process on $VLLM_NODE:$VLLM_PORT" >&2; exit 2; }

VLLM_LOG_PATH="$(
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$VLLM_NODE" \
    "readlink /proc/$VLLM_PID/fd/1" |
  awk '/^\// { print; exit }'
)"
[[ -n "$VLLM_LOG_PATH" ]] || { echo "Could not resolve vLLM log path for PID $VLLM_PID" >&2; exit 2; }

VLLM_LOG_START_SIZE="$(
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$VLLM_NODE" \
    "stat -c %s '$VLLM_LOG_PATH'" |
  awk '/^[0-9]+$/ { print; exit }'
)"
[[ -n "$VLLM_LOG_START_SIZE" ]] || { echo "Could not stat vLLM log: $VLLM_LOG_PATH" >&2; exit 2; }

echo "VLLM_PID=$VLLM_PID"
echo "VLLM_LOG_PATH=$VLLM_LOG_PATH"
echo "VLLM_LOG_START_SIZE=$VLLM_LOG_START_SIZE"

ssh -o BatchMode=yes -o ConnectTimeout=8 "$VLLM_NODE" \
  "curl --noproxy '*' -fsS -m 8 http://127.0.0.1:$VLLM_PORT/metrics" \
  >"$RUN_ROOT/vllm_metrics_before.prom"

sampler_pids=()

cleanup_samplers() {
  for pid in "${sampler_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${sampler_pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}

start_sampler() {
  local name="$1"
  local node="$2"
  local command="$3"
  local stdout_file="$4"
  local stderr_file="$5"

  echo "Starting sampler $name on $node"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$node" "$command" \
    >"$stdout_file" 2>"$stderr_file" &
  sampler_pids+=("$!")
}

start_sampler "cpu_vllm" "$VLLM_NODE" \
  "python3 '$ROOT/chemgraph_profile_cpu_sampler.py' --interval '$SAMPLE_INTERVAL' --host '$VLLM_NODE'" \
  "$RUN_ROOT/cpu_vllm.csv" "$RUN_ROOT/cpu_vllm.err"

start_sampler "gpu_vllm" "$VLLM_NODE" \
  "python3 '$ROOT/chemgraph_profile_gpu_sampler.py' --interval '$SAMPLE_INTERVAL' --host '$VLLM_NODE'" \
  "$RUN_ROOT/gpu_vllm.csv" "$RUN_ROOT/gpu_vllm.err"

start_sampler "cpu_worker" "$WORKER_NODE" \
  "python3 '$ROOT/chemgraph_profile_cpu_sampler.py' --interval '$SAMPLE_INTERVAL' --host '$WORKER_NODE'" \
  "$RUN_ROOT/cpu_worker.csv" "$RUN_ROOT/cpu_worker.err"

start_sampler "gpu_worker" "$WORKER_NODE" \
  "python3 '$ROOT/chemgraph_profile_gpu_sampler.py' --interval '$SAMPLE_INTERVAL' --host '$WORKER_NODE'" \
  "$RUN_ROOT/gpu_worker.csv" "$RUN_ROOT/gpu_worker.err"

start_sampler "vllm_metrics" "$VLLM_NODE" \
  "while true; do printf '# sample_epoch='; date +%s.%N; curl --noproxy '*' -fsS -m 8 http://127.0.0.1:$VLLM_PORT/metrics; printf '# end_sample\n'; sleep '$SAMPLE_INTERVAL'; done" \
  "$RUN_ROOT/vllm_metrics_samples.prom" "$RUN_ROOT/vllm_metrics_samples.err"

trap cleanup_samplers EXIT

PROFILE_START_EPOCH="$(date +%s.%N)"
PROFILE_START_ISO="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"

set +e
CHEMGRAPH_CONCURRENCY_LOG_ROOT="$SMOKE_ROOT" \
  VLLM_NODE="$VLLM_NODE" \
  VLLM_PORT="$VLLM_PORT" \
  CHEMGRAPH_MODEL="$MODEL" \
  "$CONCURRENCY_SCRIPT" >"$RUN_ROOT/concurrency_launcher.log" 2>&1
SMOKE_STATUS="$?"
set -e

PROFILE_END_EPOCH="$(date +%s.%N)"
PROFILE_END_ISO="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"

cleanup_samplers
trap - EXIT

ssh -o BatchMode=yes -o ConnectTimeout=8 "$VLLM_NODE" \
  "curl --noproxy '*' -fsS -m 8 http://127.0.0.1:$VLLM_PORT/metrics" \
  >"$RUN_ROOT/vllm_metrics_after.prom"

ssh -o BatchMode=yes -o ConnectTimeout=8 "$VLLM_NODE" \
  "tail -c +$((VLLM_LOG_START_SIZE + 1)) '$VLLM_LOG_PATH'" \
  >"$RUN_ROOT/vllm_backend_incremental.log"

python3 - "$RUN_ROOT/profile_metadata.json" <<PY
import json
import sys

metadata = {
    "profile_start_epoch": "$PROFILE_START_EPOCH",
    "profile_end_epoch": "$PROFILE_END_EPOCH",
    "profile_start_iso": "$PROFILE_START_ISO",
    "profile_end_iso": "$PROFILE_END_ISO",
    "smoke_status": int("$SMOKE_STATUS"),
    "run_root": "$RUN_ROOT",
    "smoke_root": "$SMOKE_ROOT",
    "worker_node": "$WORKER_NODE",
    "vllm_node": "$VLLM_NODE",
    "vllm_port": "$VLLM_PORT",
    "model": "$MODEL",
    "sample_interval_seconds": "$SAMPLE_INTERVAL",
    "vllm_pid": "$VLLM_PID",
    "vllm_log_path": "$VLLM_LOG_PATH",
    "vllm_log_start_size": "$VLLM_LOG_START_SIZE",
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2)
PY

python3 "$ROOT/chemgraph_profile_report.py" --run-root "$RUN_ROOT" \
  >"$RUN_ROOT/profile_report_stdout.log" 2>"$RUN_ROOT/profile_report_stderr.log"

echo
echo "Profile summary: $RUN_ROOT/profile_summary.md"
cat "$RUN_ROOT/profile_summary.md"

exit "$SMOKE_STATUS"
