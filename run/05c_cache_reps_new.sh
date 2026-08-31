#!/bin/bash
#SBATCH --job-name=cache_reps_new
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --array=0-3
#SBATCH --output=./logs/data_prep/%x_%A_%a.out
#SBATCH --error=./logs/data_prep/%x_%A_%a.err
#SBATCH --exclude=n0046

# ── lmsys_user + dair_emotion → Qwen3-4B layer 27 last-token reps ──────────
# 4-way DDP shard (rank 0..3, world_size=4), each rank ~10k rows
# Output: data/representations/finetune_diversified_qwen3_l27/
#         {name}_rank{r}_of_4.npy   (will coexist with old rank0_of_1.npy
#          shards from extract_diversified.py — different world_size per
#          source is OK, FinetuneDataset only requires internal consistency)
#
# Throughput @ 40G MIG: ~55 rows/sec/rank → ~4 min per rank.
# Wall clock: ~5 min (3 min compute + ~30s model load).
# ───────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

mkdir -p logs/data_prep

WORLD_SIZE=4

echo "[cache_reps_new] rank ${SLURM_ARRAY_TASK_ID}/${WORLD_SIZE}  node=${SLURMD_NODENAME}"

python experiments/data_prep/cache_finetune_reps.py \
    --rank       "${SLURM_ARRAY_TASK_ID}" \
    --world-size "${WORLD_SIZE}"           \
    --all                                   \
    --raw-dir    "${PROJ}/data/raw/finetune_new"          \
    --out-dir    "${PROJ}/data/representations/finetune_diversified_qwen3_l27" \
    --model-name "Qwen/Qwen3-4B"           \
    --layer      27                         \
    --batch-size 2000

echo "[cache_reps_new] rank ${SLURM_ARRAY_TASK_ID} done"
