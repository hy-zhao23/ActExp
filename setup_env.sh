# Activate the environment and load any cluster modules before submitting jobs.
export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
export MAMBA_ENV="${MAMBA_ENV:-oracle}"
export HF_HOME="${HF_HOME:-${PROJ}/.cache/huggingface}"
export PYTHONPATH="${PROJ}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${PROJ}/logs/cpu" "${PROJ}/logs/gpu" "${PROJ}/logs/data_prep" \
    "${PROJ}/logs/qa_gen" "${HF_HOME}"
