#!/usr/bin/env bash
set -euo pipefail

# Run on the same compute node as the local vLLM server.
# This smoke suite runs one representative sample for Exp1-Exp14.
# MACE-based cases are intentionally expressed as FAIRChem/UMA cases to avoid
# the current MACE/e3nn environment incompatibility.

ROOT="${CHEMGRAPH_LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
IMAGE="${CHEMGRAPH_CONTAINER:-$ROOT/containers/chemgraph_f31f5b1.sif}"
WORKDIR="${CHEMGRAPH_WORKDIR:-$ROOT/ChemGraph}"
HF_CACHE="${CHEMGRAPH_HF_CACHE:-$ROOT/hf_cache}"
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL="${CHEMGRAPH_MODEL:-chemgraph-qwen3-32b}"
BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:$VLLM_PORT/v1}"
PROXY="${PROXY_URL:-http://proxy.alcf.anl.gov:3128}"
NO_PROXY_LOCAL="${NO_PROXY:-127.0.0.1,localhost,::1}"
TIMEOUT_SECONDS="${CHEMGRAPH_SMOKE_TIMEOUT:-900}"
DEFAULT_RECURSION_LIMIT="${CHEMGRAPH_RECURSION_LIMIT:-20}"
REACTION_RECURSION_LIMIT="${CHEMGRAPH_REACTION_RECURSION_LIMIT:-100}"
UMA_METHOD="${CHEMGRAPH_UMA_METHOD:-FAIRChem calculator with task_name omol, model_name uma-s-1p1, device cpu}"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOGDIR="${CHEMGRAPH_SMOKE_LOG_DIR:-$WORKDIR/cg_logs/local_vllm_exp_smoke_uma_$STAMP}"

module use /soft/modulefiles
module load spack-pe-base
module load apptainer

[[ -f "$IMAGE" ]] || { echo "Missing ChemGraph image: $IMAGE" >&2; exit 2; }
[[ -d "$WORKDIR" ]] || { echo "Missing ChemGraph workdir: $WORKDIR" >&2; exit 2; }
mkdir -p "$HF_CACHE" "$LOGDIR"

# Keep local vLLM traffic off the proxy, but allow PubChem/HF requests through ALCF proxy.
curl --noproxy '*' -fsS "$BASE_URL/models" >/dev/null

ids=()
wfs=()
queries=()
recursion_limits=()

add_case() {
  ids+=("$1")
  wfs+=("$2")
  queries+=("$3")
  recursion_limits+=("${4:-$DEFAULT_RECURSION_LIMIT}")
}

add_case Exp1 single_agent \
  'Provide the SMILES string corresponding to this molecule: 9-[(2,6-dichlorophenyl)methyl]-N-(furan-2-ylmethyl)purin-6-amine'

add_case Exp2 single_agent \
  'Provide the XYZ coordinates corresponding to this molecule: benzene'

add_case Exp3 single_agent \
  "Perform geometry optimization for a molecule Hydrogen using $UMA_METHOD."

add_case Exp4 single_agent \
  "Run vibrational frequency calculation for a molecule 3-methyl-1,2,4-trithiolane using $UMA_METHOD."

add_case Exp5 single_agent \
  'Calculate the Gibbs free energy of a molecule 2-(7-methoxy-1-benzofuran-3-yl)acetic acid using GFN2-xTB at a temperature of 800K'

add_case Exp6 single_agent \
  "Perform geometry optimization for a molecule 2,3,3,3-tetrafluoropropanoic acid using $UMA_METHOD. Save the optimized coordinate in an XYZ file."

add_case Exp7 single_agent \
  'Provide the XYZ coordinates corresponding to this SMILES string: C1=CC=C(C=C1)NNC(=O)C2=NC3=CC=CC=C3C=C2'

add_case Exp8 single_agent \
  'Perform geometry optimization for this SMILES string [H][H] using NWChem, B3LYP and sto-3g'

add_case Exp9 single_agent \
  "Run vibrational frequency calculation for this SMILES string C1=CN=CC=C1C#N using $UMA_METHOD."

add_case Exp10 single_agent \
  "Calculate the Gibbs free energy of this SMILES string C1=CC=C2C(=C1)C(=NS2(=O)=O)OC3=CC=CC=C3Cl using $UMA_METHOD at T=800K."

add_case Exp11 single_agent \
  "Perform geometry optimization for this SMILES string CN1C2=C(C(=O)NC1=O)N(C(=S)N2)CCOC using $UMA_METHOD. Save the optimized coordinate in an XYZ file."

add_case Exp12 single_agent \
  'You are given a chemical reaction: 1 (Methane) + 2 (Oxygen) -> 1 (Carbon dioxide) + 2 (Water). Calculate the enthalpy for this reaction using GFN2-xTB at 400K.' \
  "$REACTION_RECURSION_LIMIT"

add_case Exp13 single_agent \
  "What is the Gibbs free energy of reaction for 1 (Methane) + 2 (Oxygen) -> 1 (Carbon dioxide) + 2 (Water) using $UMA_METHOD at 500K?"

add_case Exp14 multi_agent \
  'You are given a chemical reaction: 1 (Methane) + 2 (Oxygen) -> 1 (Carbon dioxide) + 2 (Water). Calculate the enthalpy for this reaction using GFN2-xTB at 400K.' \
  "$REACTION_RECURSION_LIMIT"

base_env=(
  --bind "$WORKDIR:/work"
  --bind "$HF_CACHE:/hf_cache"
  --pwd /work
  --env "VLLM_BASE_URL=$BASE_URL"
  --env "OPENAI_API_KEY=${OPENAI_API_KEY:-dummy_vllm_key}"
  --env "HTTP_PROXY=$PROXY" --env "HTTPS_PROXY=$PROXY"
  --env "http_proxy=$PROXY" --env "https_proxy=$PROXY"
  --env "NO_PROXY=$NO_PROXY_LOCAL" --env "no_proxy=$NO_PROXY_LOCAL"
  --env HF_HOME=/hf_cache --env HUGGINGFACE_HUB_CACHE=/hf_cache
  --env PYTHONPATH=/work/src
)

classify() {
  local code="$1"
  local log="$2"

  if [[ "$code" -eq 124 ]]; then
    echo TIMEOUT
    return
  fi

  if [[ "$code" -ne 0 ]]; then
    echo FAIL
    return
  fi

  if grep -Eq 'Traceback|Unsupported workflow type|Error running workflow|Error processing query|Recursion limit|APIConnectionError|BadRequestError|"status": "failure"|CalculationFailed|failed with command|MPI_ABORT|list index out of range|too many values to unpack' "$log"; then
    echo FAIL
    return
  fi

  echo PASS
}

run_case() {
  local workflow="$1"
  local query="$2"
  local log="$3"
  local recursion_limit="$4"

  timeout "$TIMEOUT_SECONDS" apptainer exec "${base_env[@]}" "$IMAGE" \
    chemgraph -q "$query" -m "$MODEL" -w "$workflow" -o last_message \
      --recursion-limit "$recursion_limit" >"$log" 2>&1
}

echo "LOGDIR=$LOGDIR"
echo "BASE_URL=$BASE_URL"
echo "MODEL=$MODEL"
echo "UMA_METHOD=$UMA_METHOD"
echo "TIMEOUT_SECONDS=$TIMEOUT_SECONDS"
echo "DEFAULT_RECURSION_LIMIT=$DEFAULT_RECURSION_LIMIT"
echo "REACTION_RECURSION_LIMIT=$REACTION_RECURSION_LIMIT"
printf 'id,status,seconds,workflow,recursion_limit,log\n' | tee "$LOGDIR/summary.csv"

for i in "${!ids[@]}"; do
  id="${ids[$i]}"
  workflow="${wfs[$i]}"
  query="${queries[$i]}"
  recursion_limit="${recursion_limits[$i]}"
  log="$LOGDIR/$id.log"

  printf '\n== %s (%s) ==\n' "$id" "$workflow"
  printf 'query: %s\n' "$query" >"$log.query"
  printf 'recursion_limit: %s\n' "$recursion_limit" >>"$log.query"

  start="$(date +%s)"
  set +e
  run_case "$workflow" "$query" "$log" "$recursion_limit"
  code="$?"
  set -e
  end="$(date +%s)"
  seconds="$((end - start))"
  status="$(classify "$code" "$log")"

  printf '%s,%s,%s,%s,%s,%s\n' "$id" "$status" "$seconds" "$workflow" "$recursion_limit" "$log" | tee -a "$LOGDIR/summary.csv"
  tail -n 8 "$log" | sed 's/^/  tail: /'
done

echo
echo "Summary written to $LOGDIR/summary.csv"
