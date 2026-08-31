#!/bin/bash
#SBATCH --job-name=cache_div_yi34b_40g
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:2            # 2× 40G MIG slices = 80GB total (matches 2× a100_80g mem)
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
# bf16 weights ~68GB → shards across 2× a100_40g (80GB total) via device_map=auto.
#
# Why MIG 40G slices instead of full 80G A100:
#   - The original 2× a100_80g run (1024556/1024626/1025505/1025718/1026120) produced
#     11.8M NaN values across 5 rank-1 shards (likely Yi-specific attn-mask mutation
#     bug under device_map=auto on 80G full A100s).
#   - MIG 40G slices are independent CUDA devices and far less contended (n0001/0002/
#     0089/0091 idle), giving 4 simultaneous array tasks easily.
#   - Caching script now has NaN-detection (fail-fast) so corruption is no longer silent.
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

echo "[cache_div_yi34b_40g] rank ${SLURM_ARRAY_TASK_ID}/${WORLD_SIZE}  layer=${LAYER}  GPUs=${CUDA_VISIBLE_DEVICES}  node=${SLURMD_NODENAME}"
nvidia-smi -L

python "${PROJ}/experiments/data_prep/cache_finetune_reps_vlm.py" \
    --rank       "${SLURM_ARRAY_TASK_ID}" \
    --world-size "${WORLD_SIZE}"          \
    --all                                 \
    --raw-dir    "${RAW_DIR}"             \
    --model-name "${MODEL}"               \
    --layer      "${LAYER}"               \
    --out-dir    "${OUT_DIR}"             \
    --device-map auto                     \
    --batch-size 64

echo "[cache_div_yi34b_40g] rank ${SLURM_ARRAY_TASK_ID} done"
