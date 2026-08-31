#!/bin/bash
#SBATCH --job-name=reb_cache_qwen3_base_l27
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --array=0-3
#SBATCH --output=./logs/data_prep/%x_%A_%a.out
#SBATCH --error=./logs/data_prep/%x_%A_%a.err
#SBATCH --exclude=n0046,n0004

# ── rebuttal: cache Qwen3-4B-BASE l27 reps (diversified finetune data) ─────
# Mirrors run/17_cache_diverse_qwen3_layer_ablation.sh (LAYER=27), only
# model-name → Qwen/Qwen3-4B-Base, out-dir → *_qwen3_base_l27 (新目录，
# 不动现有 instruct 版 finetune_diversified_qwen3_l27).
#
# Output: data/representations/finetune_diversified_qwen3_base_l27/
#         {source}_rank{r}_of_4.npy + metadata.json (rank 0)
# ───────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

mkdir -p logs/data_prep

WORLD_SIZE=4
OUT_DIR="${PROJ}/data/representations/finetune_diversified_qwen3_base_l27"

echo "[reb_cache_base] rank ${SLURM_ARRAY_TASK_ID}/${WORLD_SIZE}  node=${SLURMD_NODENAME}"
echo "[reb_cache_base] out_dir=${OUT_DIR}"

python experiments/data_prep/cache_finetune_reps.py \
    --rank       "${SLURM_ARRAY_TASK_ID}" \
    --world-size "${WORLD_SIZE}"           \
    --all                                   \
    --raw-dir    "${PROJ}/data/raw/finetune_diversified" \
    --out-dir    "${OUT_DIR}"               \
    --model-name "Qwen/Qwen3-4B-Base"      \
    --layer      27                         \
    --batch-size 2000

echo "[reb_cache_base] rank ${SLURM_ARRAY_TASK_ID} done"
