#!/bin/bash
#SBATCH --job-name=cache_div_gemma3_12b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1                # full 80G A100: bf16 weights ~24GB + acts
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
# Cache layer-40 last-token residual stream from google/gemma-3-12b-it
# over data/raw/finetune_diversified/*.jsonl → finetune_diversified_gemma3_12b_it_l40/
#
# Donor-swap experiment: 3× larger same-family donor.
# Layer 40 ≈ 83% depth (48 hidden layers).
# ────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

MODEL="google/gemma-3-12b-it"
LAYER=40
OUT_DIR="${PROJ}/data/representations/finetune_diversified_gemma3_12b_it_l40"
RAW_DIR="${PROJ}/data/raw/finetune_diversified"
WORLD_SIZE=8

mkdir -p "${OUT_DIR}" ./logs/data_prep

echo "[cache_div_gemma3_12b] rank ${SLURM_ARRAY_TASK_ID}/${WORLD_SIZE}  layer=${LAYER}  GPU=${CUDA_VISIBLE_DEVICES}  node=${SLURMD_NODENAME}"

python "${PROJ}/experiments/data_prep/cache_finetune_reps_vlm.py" \
    --rank       "${SLURM_ARRAY_TASK_ID}" \
    --world-size "${WORLD_SIZE}"          \
    --all                                 \
    --raw-dir    "${RAW_DIR}"             \
    --model-name "${MODEL}"               \
    --layer      "${LAYER}"               \
    --out-dir    "${OUT_DIR}"             \
    --batch-size 300

echo "[cache_div_gemma3_12b] rank ${SLURM_ARRAY_TASK_ID} done"
