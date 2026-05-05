#!/usr/bin/env bash
set -euo pipefail

# Run on the same compute node as the local vLLM server.
ROOT="${CHEMGRAPH_LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
IMAGE="${CHEMGRAPH_CONTAINER:-$ROOT/containers/chemgraph_f31f5b1.sif}"
WORKDIR="${CHEMGRAPH_WORKDIR:-$ROOT/ChemGraph}"
HF_CACHE="${CHEMGRAPH_HF_CACHE:-$ROOT/hf_cache}"
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL="${CHEMGRAPH_MODEL:-chemgraph-qwen3-32b}"
BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:$VLLM_PORT/v1}"
PROXY="${PROXY_URL:-http://proxy.alcf.anl.gov:3128}"
NO_PROXY_LOCAL="${NO_PROXY:-127.0.0.1,localhost,::1}"

module use /soft/modulefiles
module load spack-pe-base
module load apptainer

[[ -f "$IMAGE" ]] || { echo "Missing ChemGraph image: $IMAGE" >&2; exit 2; }
[[ -d "$WORKDIR" ]] || { echo "Missing ChemGraph workdir: $WORKDIR" >&2; exit 2; }
mkdir -p "$HF_CACHE"

# Keep local vLLM traffic off the proxy, but allow PubChem/HF requests through ALCF proxy.
curl --noproxy '*' -fsS "$BASE_URL/models" >/dev/null

run_chemgraph() {
  local query="$1"
  apptainer exec \
    --bind "$WORKDIR:/work" \
    --bind "$HF_CACHE:/hf_cache" \
    --pwd /work \
    --env VLLM_BASE_URL="$BASE_URL" \
    --env OPENAI_API_KEY="${OPENAI_API_KEY:-dummy_vllm_key}" \
    --env HTTP_PROXY="$PROXY" --env HTTPS_PROXY="$PROXY" \
    --env http_proxy="$PROXY" --env https_proxy="$PROXY" \
    --env NO_PROXY="$NO_PROXY_LOCAL" --env no_proxy="$NO_PROXY_LOCAL" \
    --env HF_HOME=/hf_cache --env HUGGINGFACE_HUB_CACHE=/hf_cache \
    --env PYTHONPATH=/work/src \
    "$IMAGE" chemgraph -q "$query" -m "$MODEL" -w single_agent -o last_message
}

echo "== ChemGraph default single-agent aspirin optimization =="
run_chemgraph "Optimize the geometry of an aspirin molecule."

echo "== ChemGraph best-available-tool aspirin optimization =="
run_chemgraph "Optimize the geometry of an aspirin molecule. Choose the most accurate available computational calculator for the optimization."
