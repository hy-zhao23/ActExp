#!/bin/bash
#SBATCH --job-name=cache_ft
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1              # 40G MIG — less contention + lower CU than 80G
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --output=./logs/data_prep/%x_%A_%a.out
#SBATCH --error=./logs/data_prep/%x_%A_%a.err
#SBATCH --array=0-7                        # 8-way shard
#SBATCH --exclude=n0046

# ── Stage 2 finetune activations — Llama-3.1-8B layer 27 ───────────────────
# Reads all data/raw/finetune/*.jsonl, caches to data/representations/finetune/
# 8 parallel ranks each process 1/8 of every source.
#
# Resume: per-shard .progress file → auto-resume on requeue
#
# Expected total data: ~2M texts (wiki subtypes + sci + new sources)
# Expected runtime: ~20-30 min per rank on 80G A100
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

mkdir -p logs/data_prep

WORLD_SIZE=8

echo "[cache_ft] rank ${SLURM_ARRAY_TASK_ID}/${WORLD_SIZE}  GPU=${CUDA_VISIBLE_DEVICES}  node=${SLURMD_NODENAME}"

python experiments/data_prep/cache_finetune_reps.py \
    --rank       "${SLURM_ARRAY_TASK_ID}" \
    --world-size "${WORLD_SIZE}"          \
    --all                                  \
    --batch-size 1500                      # 40G-safe (16GB model + ~4GB act)

echo "[cache_ft] rank ${SLURM_ARRAY_TASK_ID} done"
