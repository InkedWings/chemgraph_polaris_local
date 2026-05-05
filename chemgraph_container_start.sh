module use /soft/modulefiles/
module load spack-pe-base
module load apptainer

set -euo pipefail

INSTANCE_NAME="cg1"
IMAGE="./containers/chemgraph_f31f5b1.sif"
BIND_PATH="/eagle/lc-mpi/ZhijingYe/Agentic/ChemGraph:/work"

# If an old instance exists, stop it first to avoid start errors.
if apptainer instance list | awk '{print $1}' | grep -qx "${INSTANCE_NAME}"; then
  apptainer instance stop "${INSTANCE_NAME}"
fi

apptainer instance start --nv --bind "${BIND_PATH}" "${IMAGE}" "${INSTANCE_NAME}"

# export GEMINI_API_KEY="<your_gemini_api_key>"
# Pass GEMINI_API_KEY into the container if it is set in host env.
# if [[ -n "${GEMINI_API_KEY:-}" ]]; then
#   export APPTAINERENV_GEMINI_API_KEY="${GEMINI_API_KEY}"
# fi

# apptainer exec --pwd /work "instance://${INSTANCE_NAME}" \
#   chemgraph -q "Optimize aspirin molecule geometry" -m gemini-2.5-flash -w single_agent
#   chemgraph -q "Generate multiple conformers of aspirin, optimize each, and rank by energy" -m gemini-2.5-flash -w single_agent
apptainer shell instance://cg1
