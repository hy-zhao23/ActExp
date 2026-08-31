#!/bin/bash
#SBATCH --job-name=cache_ft_qwen3_l27
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --output=./logs/data_prep/%x_%A_%a.out
#SBATCH --error=./logs/data_prep/%x_%A_%a.err
#SBATCH --array=0-7
#SBATCH --exclude=n0046

# ── Stage 2 finetune activations — Qwen3-4B layer 27 (75% depth) ───────────
#   For AO self-interpretation baseline. Qwen3-4B = 36 layers, 75% → layer 27.
#   Cache at data/representations/finetune_qwen3_l27/ (separate from existing
#   finetune_qwen3 which is at layer 30).
#
#   Parallel with 05b_cache_finetune_reps_qwen3.sh; only --layer and --out-dir
#   differ. 40G MIG is plenty (Qwen3-4B bf16 ≈ 8GB, batch 2000 fits).
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

mkdir -p logs/data_prep

WORLD_SIZE=8

echo "[cache_ft_qwen3_l27] rank ${SLURM_ARRAY_TASK_ID}/${WORLD_SIZE}  GPU=${CUDA_VISIBLE_DEVICES}  node=${SLURMD_NODENAME}"

python experiments/data_prep/cache_finetune_reps.py \
    --rank       "${SLURM_ARRAY_TASK_ID}" \
    --world-size "${WORLD_SIZE}"          \
    --all                                  \
    --model-name "Qwen/Qwen3-4B"          \
    --layer      27                        \
    --batch-size 2000                      \
    --out-dir    "${PROJ}/data/representations/finetune_qwen3_l27"

echo "[cache_ft_qwen3_l27] rank ${SLURM_ARRAY_TASK_ID} done"
