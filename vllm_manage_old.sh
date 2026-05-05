#!/usr/bin/env bash
set -euo pipefail

# Manage a vLLM OpenAI-compatible server on a Polaris compute node.
# By default the target node is discovered from VLLM_PBS_JOBID.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT_DIR="${VLLM_ROOT:-$(cd "$(dirname "$SCRIPT_PATH")" && pwd)}"

VLLM_PBS_JOBID="${VLLM_PBS_JOBID:-7114910}"
VLLM_NODE="${VLLM_NODE:-auto}"
SERVICE_NAME="${VLLM_SERVICE_NAME:-vllm_qwen3_32b_8000}"
MODEL="${VLLM_MODEL:-Qwen/Qwen3-32B}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-qwen/qwen3-32b}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
CUDA_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CONTAINER="${VLLM_CONTAINER:-$ROOT_DIR/containers/vllm-openai-v0.19.1.sif}"
HF_HOME_DIR="${VLLM_HF_HOME:-$ROOT_DIR/hf_cache}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$ROOT_DIR/vllm_cache}"
LOG_DIR="${VLLM_LOG_DIR:-$ROOT_DIR/logs}"
PIDFILE="$LOG_DIR/$SERVICE_NAME.pid"
LATEST_LOG="$LOG_DIR/$SERVICE_NAME.latest.log"

BASE_SCRATCH_DIR="${BASE_SCRATCH_DIR:-/local/scratch}"
APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$BASE_SCRATCH_DIR/zye25-vllm-apptainer-tmpdir}"
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$BASE_SCRATCH_DIR/zye25-vllm-apptainer-cachedir}"

PROXY_URL="${PROXY_URL:-http://proxy.alcf.anl.gov:3128}"
HTTP_PROXY="${HTTP_PROXY:-$PROXY_URL}"
HTTPS_PROXY="${HTTPS_PROXY:-$PROXY_URL}"
http_proxy="${http_proxy:-$HTTP_PROXY}"
https_proxy="${https_proxy:-$HTTPS_PROXY}"
NO_PROXY_BASE="127.0.0.1,localhost,::1"
NO_PROXY="${NO_PROXY:-${no_proxy:-$NO_PROXY_BASE}}"
no_proxy="${no_proxy:-$NO_PROXY}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  start          Start the vLLM server if it is not already running
  status         Show process, port, model, log, and GPU status
  stop           Stop the managed vLLM server
  restart        Stop then start the server
  logs [N|-f]    Show latest log tail, or follow with -f/follow
  test           Call /v1/models and send a short chat completion request
  config         Print effective configuration

Common overrides:
  VLLM_PBS_JOBID=7114910
  VLLM_NODE=auto
  VLLM_NODE=x3208c0s7b0n0
  VLLM_PORT=8000
  VLLM_MODEL=Qwen/Qwen3-32B
  VLLM_SERVED_MODEL_NAME=qwen/qwen3-32b
  VLLM_TENSOR_PARALLEL_SIZE=4
  VLLM_MAX_MODEL_LEN=32768
  VLLM_CUDA_VISIBLE_DEVICES=0,1,2,3
  HF_TOKEN=...   Optional; forwarded into the container if set
EOF
}

quote_args() {
  local arg
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
}

forward_env_assignments() {
  local name
  local names=(
    VLLM_ROOT VLLM_PBS_JOBID VLLM_NODE VLLM_SERVICE_NAME VLLM_MODEL VLLM_SERVED_MODEL_NAME
    VLLM_HOST VLLM_PORT VLLM_TENSOR_PARALLEL_SIZE VLLM_MAX_MODEL_LEN
    VLLM_GPU_MEMORY_UTILIZATION VLLM_CUDA_VISIBLE_DEVICES VLLM_CONTAINER
    VLLM_HF_HOME VLLM_CACHE_ROOT VLLM_LOG_DIR BASE_SCRATCH_DIR
    APPTAINER_TMPDIR APPTAINER_CACHEDIR PROXY_URL HTTP_PROXY HTTPS_PROXY
    http_proxy https_proxy NO_PROXY no_proxy HF_TOKEN
  )

  for name in "${names[@]}"; do
    if [[ ${!name+x} ]]; then
      printf ' %s=%q' "$name" "${!name}"
    fi
  done
}

host_matches_target() {
  local short fqdn
  short="$(hostname -s 2>/dev/null || hostname)"
  fqdn="$(hostname -f 2>/dev/null || hostname)"

  [[ "$VLLM_NODE" == "local" ]] && return 0
  [[ "$short" == "$VLLM_NODE" ]] && return 0
  [[ "$fqdn" == "$VLLM_NODE" ]] && return 0
  [[ "$fqdn" == "$VLLM_NODE."* ]] && return 0
  return 1
}

discover_vllm_node() {
  if [[ "$VLLM_NODE" != "auto" ]]; then
    printf '%s\n' "$VLLM_NODE"
    return 0
  fi

  if [[ -n "${PBS_NODEFILE:-}" && -r "${PBS_NODEFILE:-}" ]]; then
    sort -u "$PBS_NODEFILE" | head -n 1
    return 0
  fi

  local jobid="$VLLM_PBS_JOBID"
  if [[ -z "$jobid" || "$jobid" == "auto" ]]; then
    jobid="$(qstat -u "$USER" 2>/dev/null | awk 'NR > 5 && $10 == "R" {print $1; exit}' | cut -d. -f1)"
  fi

  if [[ -z "$jobid" ]]; then
    echo "Could not discover a running PBS job. Set VLLM_PBS_JOBID or VLLM_NODE." >&2
    return 1
  fi

  qstat -n "$jobid" 2>/dev/null \
    | tr '+[:space:]' '\n' \
    | sed -n 's#^\(x[0-9][^/]*\)/.*#\1#p' \
    | head -n 1
}

resolve_vllm_node() {
  if [[ "$VLLM_NODE" == "auto" ]]; then
    VLLM_NODE="$(discover_vllm_node)"
    if [[ -z "$VLLM_NODE" ]]; then
      echo "Could not resolve VLLM_NODE from PBS job $VLLM_PBS_JOBID." >&2
      exit 1
    fi
  fi
}

maybe_run_remote() {
  local cmd="$1"
  shift || true

  resolve_vllm_node

  if [[ "${LOCAL_MODE:-0}" == "1" ]] || host_matches_target; then
    return 0
  fi

  local remote_cmd
  remote_cmd="env$(forward_env_assignments) bash $(printf '%q' "$SCRIPT_PATH") --local $(printf '%q' "$cmd")$(quote_args "$@")"
  exec ssh "$VLLM_NODE" "$remote_cmd"
}

setup_polaris_env() {
  ml use /soft/modulefiles
  ml spack-pe-base
  ml apptainer

  export BASE_SCRATCH_DIR APPTAINER_TMPDIR APPTAINER_CACHEDIR
  export HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

  mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$HF_HOME_DIR" "$VLLM_CACHE_ROOT" "$LOG_DIR"
}

current_pid() {
  [[ -s "$PIDFILE" ]] || return 1
  local pid
  pid="$(tr -d '[:space:]' < "$PIDFILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  printf '%s\n' "$pid"
}

vllm_pids() {
  ps -u "$USER" -o pid=,args= \
    | awk -v model="$MODEL" -v served="$SERVED_MODEL_NAME" -v port="$PORT" '
        index($0, "vllm serve") &&
        (index($0, model) || index($0, served)) &&
        index($0, "--port " port) {
          print $1
        }
      '
}

parent_chain() {
  local pid="$1"
  while [[ "$pid" =~ ^[0-9]+$ && "$pid" != "1" ]]; do
    printf '%s\n' "$pid"
    pid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d '[:space:]' || true)"
  done
}

managed_pids() {
  local pid
  for pid in $(vllm_pids); do
    parent_chain "$pid"
  done | sort -n -u
}

current_log() {
  if [[ -e "$LATEST_LOG" ]]; then
    printf '%s\n' "$LATEST_LOG"
    return 0
  fi

  local log
  log="$(ls -t "$LOG_DIR/${SERVICE_NAME}"_*.log 2>/dev/null | head -n 1 || true)"
  [[ -n "$log" ]] || return 1
  printf '%s\n' "$log"
}

port_listening() {
  ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]$PORT$"
}

run_server() {
  setup_polaris_env

  if [[ ! -f "$CONTAINER" ]]; then
    echo "Missing container: $CONTAINER" >&2
    exit 2
  fi

  exec apptainer exec --nv \
    --bind /lus/eagle:/lus/eagle,/local/scratch:/local/scratch \
    --env HF_HOME="$HF_HOME_DIR" \
    --env HUGGINGFACE_HUB_CACHE="$HF_HOME_DIR" \
    --env VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
    --env HTTP_PROXY="$HTTP_PROXY" \
    --env HTTPS_PROXY="$HTTPS_PROXY" \
    --env http_proxy="$http_proxy" \
    --env https_proxy="$https_proxy" \
    --env NO_PROXY="$NO_PROXY" \
    --env no_proxy="$no_proxy" \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    "$CONTAINER" \
    vllm serve "$MODEL" \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --trust-remote-code
}

start_server() {
  mkdir -p "$LOG_DIR"

  local pids pid
  pids="$(vllm_pids | xargs echo)"
  if [[ -n "$pids" ]]; then
    echo "Already running: PID(s)=$pids"
    status_server
    return 0
  fi

  if [[ -e "$PIDFILE" ]]; then
    echo "Removing stale pidfile: $PIDFILE"
    rm -f "$PIDFILE"
  fi

  if [[ ! -f "$CONTAINER" ]]; then
    echo "Missing container: $CONTAINER" >&2
    echo "Expected vLLM v0.19.1 SIF. Build or pull it first." >&2
    exit 2
  fi

  if port_listening; then
    echo "Port $PORT is already listening; refusing to start another server." >&2
    ss -ltnp 2>/dev/null | grep "[:.]$PORT" || true
    exit 3
  fi

  local log
  log="$LOG_DIR/${SERVICE_NAME}_$(date +%Y%m%d_%H%M%S).log"

  if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash -lc "exec $(printf '%q' "$SCRIPT_PATH") --local run-server" > "$log" 2>&1 &
  else
    nohup bash -lc "exec $(printf '%q' "$SCRIPT_PATH") --local run-server" > "$log" 2>&1 &
  fi

  pid="$!"
  printf '%s\n' "$pid" > "$PIDFILE"
  ln -sfn "$log" "$LATEST_LOG"

  echo "Started vLLM"
  echo "  node: $VLLM_NODE"
  echo "  pid:  $pid"
  echo "  url:  http://$HOST:$PORT"
  echo "  log:  $log"
}

kill_tree() {
  local pid="$1"
  local child
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    kill_tree "$child"
  done
  kill "$pid" 2>/dev/null || true
}

stop_server() {
  local pids pid
  pids="$(managed_pids | xargs echo)"
  if [[ -z "$pids" ]]; then
    echo "Not running"
    rm -f "$PIDFILE"
    return 0
  fi

  echo "Stopping PID(s)=$pids"
  kill -TERM $pids 2>/dev/null || true

  local i
  for i in {1..20}; do
    if [[ -z "$(vllm_pids | xargs echo)" ]]; then
      rm -f "$PIDFILE"
      echo "Stopped"
      return 0
    fi
    sleep 0.5
  done

  echo "Still running after TERM; sending KILL"
  pids="$(managed_pids | xargs echo)"
  [[ -n "$pids" ]] && kill -KILL $pids 2>/dev/null || true
  rm -f "$PIDFILE"
  echo "Stopped"
}

status_server() {
  echo "Service: $SERVICE_NAME"
  echo "Node:    $(hostname -f 2>/dev/null || hostname)"
  echo "Model:   $MODEL as $SERVED_MODEL_NAME"
  echo "URL:     http://127.0.0.1:$PORT"

  local pids log
  pids="$(vllm_pids | xargs echo)"
  if [[ -n "$pids" ]]; then
    echo "State:   running"
    echo "PID(s):  $pids"
    ps -o pid,ppid,stat,etime,pcpu,pmem,cmd -p "$(echo "$pids" | tr ' ' ',')" || true
  else
    echo "State:   stopped"
  fi

  if log="$(current_log)"; then
    echo "Log:     $log"
  else
    echo "Log:     none"
  fi

  if port_listening; then
    echo "Port:    $PORT listening"
    curl --noproxy '*' -s --max-time 3 "http://127.0.0.1:$PORT/v1/models" | python3 -m json.tool 2>/dev/null || true
  else
    echo "Port:    $PORT not listening"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
  fi
}

show_logs() {
  local mode="${1:-120}"
  local log
  if ! log="$(current_log)"; then
    echo "No log found under $LOG_DIR for $SERVICE_NAME" >&2
    exit 1
  fi

  if [[ "$mode" == "-f" || "$mode" == "follow" ]]; then
    tail -f "$log"
  else
    tail -n "$mode" "$log"
  fi
}

test_server() {
  if ! port_listening; then
    echo "Port $PORT is not listening" >&2
    exit 1
  fi

  echo "Models:"
  curl --noproxy '*' -sS --max-time 10 "http://127.0.0.1:$PORT/v1/models" | python3 -m json.tool

  echo
  echo "Chat test:"
  local payload response
  payload="$(python3 - <<PY
import json
print(json.dumps({
    "model": "$SERVED_MODEL_NAME",
    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    "max_tokens": 32,
}))
PY
)"
  response="$(curl --noproxy '*' -sS --max-time 120 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$payload")"
  printf '%s' "$response" | python3 -c 'import json, sys; data = json.load(sys.stdin); print(data["choices"][0]["message"]["content"])'
}

print_config() {
  cat <<EOF
VLLM_NODE=$VLLM_NODE
SERVICE_NAME=$SERVICE_NAME
ROOT_DIR=$ROOT_DIR
CONTAINER=$CONTAINER
MODEL=$MODEL
SERVED_MODEL_NAME=$SERVED_MODEL_NAME
HOST=$HOST
PORT=$PORT
TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE
MAX_MODEL_LEN=$MAX_MODEL_LEN
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
HF_HOME=$HF_HOME_DIR
VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT
LOG_DIR=$LOG_DIR
PIDFILE=$PIDFILE
APPTAINER_TMPDIR=$APPTAINER_TMPDIR
APPTAINER_CACHEDIR=$APPTAINER_CACHEDIR
EOF
}

LOCAL_MODE=0
if [[ "${1:-}" == "--local" ]]; then
  LOCAL_MODE=1
  shift
fi

cmd="${1:-status}"
shift || true

maybe_run_remote "$cmd" "$@"

case "$cmd" in
  start) start_server "$@" ;;
  status) status_server "$@" ;;
  stop) stop_server "$@" ;;
  restart) stop_server; start_server ;;
  logs) show_logs "$@" ;;
  follow) show_logs -f ;;
  test) test_server "$@" ;;
  config) print_config ;;
  run-server) run_server ;;
  help|-h|--help) usage ;;
  *)
    usage >&2
    exit 1
    ;;
esac
