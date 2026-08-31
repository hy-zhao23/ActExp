#!/bin/bash
#SBATCH --job-name=cache_gemma4_e4b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1            # 40G MIG slice — Gemma-4 E4B bf16 ≈ 8GB + activations
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --output=./logs/data_prep/%x_%A_%a.out
#SBATCH --error=./logs/data_prep/%x_%A_%a.err
#SBATCH --array=0-7
#SBATCH --exclude=n0046

# ────────────────────────────────────────────────────────────
# Cache layer-<L> last-token residual stream from google/gemma-4-E4B-it
# over data/raw/finetune/*.jsonl → data/representations/finetune_gemma4_e4b/
#
# Layer selected by matching Llama depth ratio:  L = round(num_hidden_layers * 27/32)
# Computed dynamically inside the job (see preflight below).
#
# Gemma 4 E4B is new (post-Jan-2026 release); preflight verifies the hook path
# `model.model.layers[L]` actually exists and produces a bf16 residual stream
# before committing the ~16-source cache run.
# ────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"

MODEL="google/gemma-4-E4B-it"
OUT_DIR="${PROJ}/data/representations/finetune_gemma4_e4b"
WORLD_SIZE=8

mkdir -p "${OUT_DIR}" ./logs/data_prep

# each rank computes layer independently (cheap AutoConfig read; no inter-rank wait)
# multimodal configs expose the LM sub-config under .text_config — handled via fallback
read LAYER NL HD < <(python - <<PY
from transformers import AutoConfig
c = AutoConfig.from_pretrained("${MODEL}", trust_remote_code=True)
nl = getattr(c, "num_hidden_layers", None) or c.text_config.num_hidden_layers
hd = getattr(c, "hidden_size",        None) or c.text_config.hidden_size
print(round(nl * 27/32), nl, hd)
PY
)
echo "[rank ${SLURM_ARRAY_TASK_ID}/${WORLD_SIZE}] model=${MODEL}  n_layers=${NL}  layer=${LAYER}  hidden=${HD}"

python "${PROJ}/experiments/data_prep/cache_finetune_reps_vlm.py" \
    --rank       "${SLURM_ARRAY_TASK_ID}" \
    --world-size "${WORLD_SIZE}"          \
    --all                                 \
    --model-name "${MODEL}"               \
    --layer      "${LAYER}"               \
    --out-dir    "${OUT_DIR}"             \
    --batch-size 600                      # 40G MIG; Gemma-4 E4B VLM ~12GB weights

echo "[rank ${SLURM_ARRAY_TASK_ID}] done"
