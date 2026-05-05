#!/usr/bin/env bash
set -euo pipefail

ROOT="${VLLM_ROOT:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)}"
MANAGE="${VLLM_MANAGE:-$ROOT/vllm_manage.sh}"
CLIENT="${VLLM_BENCH_CLIENT:-$ROOT/vllm_bench_client.py}"
HOST_PYTHON="${HOST_PYTHON:-/soft/applications/conda/2025-09-25/mconda3/bin/python}"
JOBID="${VLLM_PBS_JOBID:-7114910}"
NODE="${VLLM_NODE:-auto}"
MODEL="${VLLM_MODEL:-Qwen/Qwen3-32B}"
SERVED="${VLLM_SERVED_MODEL_NAME:-qwen/qwen3-32b}"
PORT="${VLLM_PORT:-8000}"
SIF="${VLLM_CONTAINER:-$ROOT/containers/vllm-openai-v0.19.1.sif}"
HF_HOME_DIR="${VLLM_HF_HOME:-$ROOT/hf_cache}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$ROOT/vllm_cache}"

TP_VALUES="${TP_VALUES:-1 2 4}"
INPUT_LENS="${INPUT_LENS:-1024 2048 4096 8192 12000 16000 20000}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
MAX_CONCURRENCY_VALUES="${MAX_CONCURRENCY_VALUES:-${MAX_CONCURRENCY:-1 2 4 8}}"
PROMPT_MODE="${PROMPT_MODE:-auto}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-24576}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-hermes}"
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-0}"
RESULT_DIR="${RESULT_DIR:-$ROOT/bench_results/qwen3_32b_$(date +%Y%m%d_%H%M%S)}"
CSV="$RESULT_DIR/summary.csv"

PROXY="${PROXY_URL:-http://proxy.alcf.anl.gov:3128}"

usage() {
  cat <<EOF
Usage: $(basename "$0")

Environment overrides:
  TP_VALUES="1 2 4"
  INPUT_LENS="1024 2048 4096 8192 12000 16000 20000"
  OUTPUT_LEN=256
  NUM_PROMPTS=16
  MAX_CONCURRENCY_VALUES="1 2 4 8"
  PROMPT_MODE=auto|token_ids|string
  READY_TIMEOUT=1200
  RESULT_DIR=$RESULT_DIR

This script restarts the vLLM server for each TP value.
EOF
}

discover_node() {
  [[ "$NODE" != "auto" ]] && { echo "$NODE"; return; }
  qstat -n "$JOBID" 2>/dev/null \
    | tr '+[:space:]' '\n' \
    | sed -n 's#^\(x[0-9][^/]*\)/.*#\1#p' \
    | head -n 1
}

devices_for_tp() {
  case "$1" in
    1) echo "0" ;;
    2) echo "0,1" ;;
    4) echo "0,1,2,3" ;;
    *) echo "Unsupported TP=$1" >&2; return 1 ;;
  esac
}

manage() {
  local tp="$1" devices="$2" cmd="$3"
  env \
    VLLM_NODE="$NODE" \
    VLLM_MODEL="$MODEL" \
    VLLM_SERVED_MODEL_NAME="$SERVED" \
    VLLM_PORT="$PORT" \
    VLLM_TENSOR_PARALLEL_SIZE="$tp" \
    VLLM_CUDA_VISIBLE_DEVICES="$devices" \
    VLLM_MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    VLLM_GPU_MEMORY_UTILIZATION="$GPU_UTIL" \
    VLLM_ENABLE_AUTO_TOOL_CHOICE="$ENABLE_AUTO_TOOL_CHOICE" \
    VLLM_TOOL_CALL_PARSER="$TOOL_CALL_PARSER" \
    VLLM_ENFORCE_EAGER="$ENFORCE_EAGER" \
    VLLM_ENABLE_PREFIX_CACHING="$ENABLE_PREFIX_CACHING" \
    VLLM_LOG_STATS_INTERVAL=1 \
    "$MANAGE" "$cmd"
}

wait_ready() {
  local tries=$((READY_TIMEOUT / 5)) i
  (( tries > 0 )) || tries=1
  for i in $(seq 1 "$tries"); do
    if ssh "$NODE" "curl --noproxy '*' -fsS http://127.0.0.1:$PORT/v1/models >/dev/null" 2>/dev/null; then
      return 0
    fi
    if (( i >= 12 )) && ! ssh "$NODE" "ps -u '$USER' -o args= | grep -F 'vllm serve' | grep -F -- '--port $PORT' | grep -F '$MODEL' | grep -v grep >/dev/null" 2>/dev/null; then
      return 1
    fi
    sleep 5
  done
  return 1
}

bench_one() {
  local input_len="$1" max_concurrency="$2" json_path="$3"
  ssh "$NODE" "ml use /soft/modulefiles; ml spack-pe-base; ml apptainer; \
    export HTTP_PROXY=$PROXY HTTPS_PROXY=$PROXY http_proxy=$PROXY https_proxy=$PROXY; \
    export NO_PROXY=127.0.0.1,localhost,::1 no_proxy=127.0.0.1,localhost,::1; \
    apptainer exec --bind /lus/eagle:/lus/eagle,/local/scratch:/local/scratch \
      --env HF_HOME=$HF_HOME_DIR \
      --env HUGGINGFACE_HUB_CACHE=$HF_HOME_DIR \
      --env VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT \
      --env NO_PROXY=127.0.0.1,localhost,::1 \
      --env no_proxy=127.0.0.1,localhost,::1 \
      --env HF_TOKEN=${HF_TOKEN:-} \
      $SIF \
      python3 $CLIENT \
        --host 127.0.0.1 \
        --port $PORT \
        --model $SERVED \
        --tokenizer-model $MODEL \
        --input-len $input_len \
        --output-len $OUTPUT_LEN \
        --num-prompts $NUM_PROMPTS \
        --max-concurrency $max_concurrency \
        --prompt-mode $PROMPT_MODE \
        --result-json $json_path"
}

write_row() {
  "$HOST_PYTHON" - "$CSV" "$@" <<'PY'
import csv, json, os, sys

csv_path, status, tp, input_len, output_len, num_prompts, max_conc, client_json, client_log, server_json, server_log, err = sys.argv[1:]
fields = [
    "status", "tp", "input_len", "output_len", "num_prompts", "max_concurrency",
    "server_prefill_toks_per_s_mean", "server_prefill_toks_per_s_max",
    "server_decode_toks_per_s_mean", "server_decode_toks_per_s_max",
    "server_stats_samples", "client_request_throughput",
    "client_input_toks_per_s", "client_output_toks_per_s",
    "client_total_toks_per_s", "actual_input_tokens_mean",
    "actual_output_tokens_mean", "mean_ttft_ms", "p50_ttft_ms",
    "p99_ttft_ms", "mean_tpot_ms", "p50_tpot_ms", "p99_tpot_ms",
    "client_json_path", "client_log_path", "server_stats_json_path",
    "server_stats_log_path", "error",
]
row = {k: "" for k in fields}
row.update({
    "status": status,
    "tp": tp,
    "input_len": input_len,
    "output_len": output_len,
    "num_prompts": num_prompts,
    "max_concurrency": max_conc,
    "client_json_path": client_json,
    "client_log_path": client_log,
    "server_stats_json_path": server_json,
    "server_stats_log_path": server_log,
    "error": err,
})

if client_json and os.path.exists(client_json):
    with open(client_json) as f:
        data = json.load(f)

    def get(*names):
        for name in names:
            if name in data and data[name] is not None:
                return data[name]
        return ""

    row["client_request_throughput"] = get("request_throughput")
    row["client_input_toks_per_s"] = get("client_input_toks_per_s")
    row["client_output_toks_per_s"] = get("client_output_toks_per_s")
    row["client_total_toks_per_s"] = get("client_total_toks_per_s")
    row["actual_input_tokens_mean"] = get("actual_input_tokens_mean")
    row["actual_output_tokens_mean"] = get("actual_output_tokens_mean")
    row["mean_ttft_ms"] = get("mean_ttft_ms")
    row["p50_ttft_ms"] = get("p50_ttft_ms")
    row["p99_ttft_ms"] = get("p99_ttft_ms")
    row["mean_tpot_ms"] = get("mean_tpot_ms")
    row["p50_tpot_ms"] = get("p50_tpot_ms")
    row["p99_tpot_ms"] = get("p99_tpot_ms")

if server_json and os.path.exists(server_json):
    with open(server_json) as f:
        stats = json.load(f)
    row["server_prefill_toks_per_s_mean"] = stats.get("prefill_toks_per_s_mean_nonzero", "")
    row["server_prefill_toks_per_s_max"] = stats.get("prefill_toks_per_s_max", "")
    row["server_decode_toks_per_s_mean"] = stats.get("decode_toks_per_s_mean_nonzero", "")
    row["server_decode_toks_per_s_max"] = stats.get("decode_toks_per_s_max", "")
    row["server_stats_samples"] = stats.get("samples", "")

new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    if new:
        writer.writeheader()
    writer.writerow(row)
PY
}

extract_server_stats() {
  local src_log="$1" start_line="$2" out_json="$3" out_log="$4"
  if [[ -n "$src_log" && -f "$src_log" ]]; then
    tail -n +"$((start_line + 1))" "$src_log" > "$out_log" || true
  else
    : > "$out_log"
  fi

  "$HOST_PYTHON" - "$out_log" "$out_json" <<'PY'
import json, os, re, statistics, sys

log_path, json_path = sys.argv[1:]
pattern = re.compile(
    r"Avg prompt throughput:\s*([0-9.]+)\s*tokens/s,\s*"
    r"Avg generation throughput:\s*([0-9.]+)\s*tokens/s"
)
rows = []
with open(log_path, errors="replace") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            rows.append({
                "prefill_toks_per_s": float(match.group(1)),
                "decode_toks_per_s": float(match.group(2)),
                "line": line.strip(),
            })

def mean(vals):
    return statistics.fmean(vals) if vals else None

prefill = [row["prefill_toks_per_s"] for row in rows]
decode = [row["decode_toks_per_s"] for row in rows]
summary = {
    "samples": len(rows),
    "prefill_toks_per_s_mean_nonzero": mean([x for x in prefill if x > 0]),
    "prefill_toks_per_s_max": max(prefill) if prefill else None,
    "decode_toks_per_s_mean_nonzero": mean([x for x in decode if x > 0]),
    "decode_toks_per_s_max": max(decode) if decode else None,
    "rows": rows,
}
os.makedirs(os.path.dirname(json_path), exist_ok=True)
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
PY
}

main() {
  [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }
  NODE="$(discover_node)"
  [[ -n "$NODE" ]] || { echo "Could not resolve VLLM node. Set VLLM_NODE." >&2; exit 1; }

  mkdir -p "$RESULT_DIR"
  {
    echo "node=$NODE"
    echo "model=$MODEL"
    echo "served_model=$SERVED"
    echo "port=$PORT"
    echo "tp_values=$TP_VALUES"
    echo "input_lens=$INPUT_LENS"
    echo "output_len=$OUTPUT_LEN"
    echo "num_prompts=$NUM_PROMPTS"
    echo "max_concurrency_values=$MAX_CONCURRENCY_VALUES"
    echo "prompt_mode=$PROMPT_MODE"
    echo "ready_timeout=$READY_TIMEOUT"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "gpu_memory_utilization=$GPU_UTIL"
    echo "enable_auto_tool_choice=$ENABLE_AUTO_TOOL_CHOICE"
    echo "tool_call_parser=$TOOL_CALL_PARSER"
    echo "enforce_eager=$ENFORCE_EAGER"
    echo "enable_prefix_caching=$ENABLE_PREFIX_CACHING"
    echo "result_dir=$RESULT_DIR"
  } > "$RESULT_DIR/config.txt"

  for tp in $TP_VALUES; do
    devices="$(devices_for_tp "$tp")"
    echo "=== TP=$tp devices=$devices ==="
    manage "$tp" "$devices" stop || true

    if ! manage "$tp" "$devices" start > "$RESULT_DIR/start_tp${tp}.log" 2>&1; then
      for input_len in $INPUT_LENS; do
        for max_concurrency in $MAX_CONCURRENCY_VALUES; do
          write_row failed "$tp" "$input_len" "$OUTPUT_LEN" "$NUM_PROMPTS" "$max_concurrency" "" "$RESULT_DIR/start_tp${tp}.log" "" "" "server_start_failed"
        done
      done
      continue
    fi

    server_log="$(awk '/^Log:/{print $2}' "$RESULT_DIR/start_tp${tp}.log" | tail -n 1)"
    [[ -n "$server_log" ]] || server_log="$ROOT/logs/vllm_qwen3_32b_8000.latest.log"

    if ! wait_ready; then
      for input_len in $INPUT_LENS; do
        for max_concurrency in $MAX_CONCURRENCY_VALUES; do
          write_row failed "$tp" "$input_len" "$OUTPUT_LEN" "$NUM_PROMPTS" "$max_concurrency" "" "$RESULT_DIR/start_tp${tp}.log" "" "" "server_not_ready"
        done
      done
      manage "$tp" "$devices" stop || true
      continue
    fi

    for input_len in $INPUT_LENS; do
      for max_concurrency in $MAX_CONCURRENCY_VALUES; do
        json_path="$RESULT_DIR/tp${tp}_in${input_len}_out${OUTPUT_LEN}_c${max_concurrency}.client.json"
        log_path="$RESULT_DIR/tp${tp}_in${input_len}_out${OUTPUT_LEN}_c${max_concurrency}.client.log"
        stats_json="$RESULT_DIR/tp${tp}_in${input_len}_out${OUTPUT_LEN}_c${max_concurrency}.server_stats.json"
        stats_log="$RESULT_DIR/tp${tp}_in${input_len}_out${OUTPUT_LEN}_c${max_concurrency}.server_stats.log"
        start_line=0
        [[ -f "$server_log" ]] && start_line="$(wc -l < "$server_log")"
        echo "TP=$tp input_len=$input_len output_len=$OUTPUT_LEN concurrency=$max_concurrency"
        if bench_one "$input_len" "$max_concurrency" "$json_path" > "$log_path" 2>&1; then
          bench_status=ok
          bench_error=""
        else
          bench_status=failed
          bench_error="benchmark_failed"
        fi
        sleep 2
        extract_server_stats "$server_log" "$start_line" "$stats_json" "$stats_log"
        write_row "$bench_status" "$tp" "$input_len" "$OUTPUT_LEN" "$NUM_PROMPTS" "$max_concurrency" "$json_path" "$log_path" "$stats_json" "$stats_log" "$bench_error"
      done
    done

    manage "$tp" "$devices" stop || true
  done

  echo "Summary: $CSV"
}

main "$@"
