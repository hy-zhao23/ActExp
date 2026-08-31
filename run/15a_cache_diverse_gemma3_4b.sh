#!/bin/bash
#SBATCH --job-name=cache_div_gemma3_4b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1            # 40G MIG: bf16 weights ~8GB + acts
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --qos=standard
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=./logs/data_prep/%x_%A_%a.out
#SBATCH --error=./logs/data_prep/%x_%A_%a.err
#SBATCH --array=0-7
#SBATCH --exclude=n0046

# ────────────────────────────────────────────────────────────
# Cache layer-28 last-token residual stream from google/gemma-3-4b-it
# over data/raw/finetune_diversified/*.jsonl → finetune_diversified_gemma3_4b_it_l28/
#
# Donor-swap baseline: same-scale (4B) cross-family donor vs Qwen3-4B target.
# Layer 28 ≈ 84% depth (34 hidden layers); matches Llama-8B@27 convention.
# ────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

MODEL="google/gemma-3-4b-it"
LAYER=28
OUT_DIR="${PROJ}/data/representations/finetune_diversified_gemma3_4b_it_l28"
RAW_DIR="${PROJ}/data/raw/finetune_diversified"
WORLD_SIZE=8

mkdir -p "${OUT_DIR}" ./logs/data_prep

echo "[cache_div_gemma3_4b] rank ${SLURM_ARRAY_TASK_ID}/${WORLD_SIZE}  layer=${LAYER}  GPU=${CUDA_VISIBLE_DEVICES}  node=${SLURMD_NODENAME}"

python "${PROJ}/experiments/data_prep/cache_finetune_reps_vlm.py" \
    --rank       "${SLURM_ARRAY_TASK_ID}" \
    --world-size "${WORLD_SIZE}"          \
    --all                                 \
    --raw-dir    "${RAW_DIR}"             \
    --model-name "${MODEL}"               \
    --layer      "${LAYER}"               \
    --out-dir    "${OUT_DIR}"             \
    --batch-size 800

echo "[cache_div_gemma3_4b] rank ${SLURM_ARRAY_TASK_ID} done"
