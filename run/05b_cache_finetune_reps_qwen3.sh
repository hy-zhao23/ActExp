#!/bin/bash
#SBATCH --job-name=cache_ft_qwen3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1              # 40G MIG, qos=low = free
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --output=./logs/data_prep/%x_%A_%a.out
#SBATCH --error=./logs/data_prep/%x_%A_%a.err
#SBATCH --array=0-7
#SBATCH --exclude=n0046

# ── Stage 2 finetune activations — Qwen3-4B layer 30 (~83% depth) ──────────
#   Parallel pair of 05_cache_finetune_reps.sh (Llama). Same JSONL sources;
#   different source model → different activation distribution. Used to compare
#   adapter quality between Llama-derived and Qwen3-derived activations.
#
# Output: data/representations/finetune_qwen3/{source}_rank{r}_of_8.npy
#
# Qwen3-4B: 36 layers, hidden_dim=2560. Layer 30 ≈ 83% depth (matches Llama 27/32).
# Memory: Qwen3-4B bf16 ≈ 8GB + batch activations → well within 40G.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

mkdir -p logs/data_prep

WORLD_SIZE=8

echo "[cache_ft_qwen3] rank ${SLURM_ARRAY_TASK_ID}/${WORLD_SIZE}  GPU=${CUDA_VISIBLE_DEVICES}  node=${SLURMD_NODENAME}"

python experiments/data_prep/cache_finetune_reps.py \
    --rank       "${SLURM_ARRAY_TASK_ID}" \
    --world-size "${WORLD_SIZE}"          \
    --all                                  \
    --model-name "Qwen/Qwen3-4B"          \
    --layer      30                        \
    --batch-size 2000                      \
    --out-dir    "${PROJ}/data/representations/finetune_qwen3"

echo "[cache_ft_qwen3] rank ${SLURM_ARRAY_TASK_ID} done"
