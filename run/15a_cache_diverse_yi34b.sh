#!/bin/bash
#SBATCH --job-name=cache_div_yi34b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2                # 2× 80G A100 for 34B bf16 (~68GB) + acts
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --qos=standard
#SBATCH --time=09:30:00
#SBATCH --requeue
#SBATCH --output=./logs/data_prep/%x_%A_%a.out
#SBATCH --error=./logs/data_prep/%x_%A_%a.err
#SBATCH --array=0-7
#SBATCH --exclude=n0046,n0004

# ────────────────────────────────────────────────────────────
# Cache layer-50 last-token residual stream from 01-ai/Yi-1.5-34B-Chat
# over data/raw/finetune_diversified/*.jsonl → finetune_diversified_yi15_34b_chat_l50/
#
# Donor-swap experiment: largest cross-family donor (~9× target params).
# Layer 50 ≈ 83% depth (60 hidden layers).
# bf16 weights ~68GB → shards across 2× A100 80G via device_map=auto.
# ────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

MODEL="01-ai/Yi-1.5-34B-Chat"
LAYER=50
OUT_DIR="${PROJ}/data/representations/finetune_diversified_yi15_34b_chat_l50"
RAW_DIR="${PROJ}/data/raw/finetune_diversified"
WORLD_SIZE=8

mkdir -p "${OUT_DIR}" ./logs/data_prep

echo "[cache_div_yi34b] rank ${SLURM_ARRAY_TASK_ID}/${WORLD_SIZE}  layer=${LAYER}  GPUs=${CUDA_VISIBLE_DEVICES}  node=${SLURMD_NODENAME}"

python "${PROJ}/experiments/data_prep/cache_finetune_reps_vlm.py" \
    --rank       "${SLURM_ARRAY_TASK_ID}" \
    --world-size "${WORLD_SIZE}"          \
    --all                                 \
    --raw-dir    "${RAW_DIR}"             \
    --model-name "${MODEL}"               \
    --layer      "${LAYER}"               \
    --out-dir    "${OUT_DIR}"             \
    --device-map auto                     \
    --batch-size 128

echo "[cache_div_yi34b] rank ${SLURM_ARRAY_TASK_ID} done"
