#!/bin/bash
#SBATCH --job-name=frozen_qwen3self_to_llama8b_fullqa
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --time=20:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004

# ── Frozen-decoder remap experiment ───────────────────────────────────────
#   Decoder (frozen):  qwen3-4b LM + LoRA from self-donor stage-2 ckpt
#                      checkpoints/oracle_ft_qwen3_4b_l27_diversified/final
#   New donor:         Llama-3.1-8B (reps under finetune_diversified/)
#   Trainable:         cross-attn adapter (fresh, random init)
#   Data:              full diversified QA (~1.13M), fullqa stage-2 format
#
#   Wall budget: ~16h on 4× A100 (eff batch 1536, ~7340 opt steps).
#   Dependency: stage-2 self-donor ft must finish first (sbatch --dependency).
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

CONFIG="experiments/training/configs/frozen_decoder_qwen3_4b_self_to_llama8b_fullqa.yaml"
DECODER_CKPT="checkpoints/oracle_ft_qwen3_4b_l27_diversified/final"

if [[ ! -d "$DECODER_CKPT" ]]; then
    echo "[ERROR] decoder ckpt missing: $DECODER_CKPT"
    echo "  available under checkpoints/oracle_ft_qwen3_4b_l27_diversified/:"
    ls checkpoints/oracle_ft_qwen3_4b_l27_diversified/ 2>&1 || true
    exit 1
fi
if [[ ! -f "${DECODER_CKPT}/lora/adapter_model.safetensors" ]]; then
    echo "[ERROR] decoder ckpt has no LoRA: ${DECODER_CKPT}/lora/"
    ls "${DECODER_CKPT}/lora/" 2>&1 || true
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[frozen_qwen3self_to_llama8b] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[frozen_qwen3self_to_llama8b] decoder_ckpt=${DECODER_CKPT}"

unset SLURM_TRES_PER_TASK

if [[ ! -d "data/raw/finetune_qa_diversified" || -z "$(ls -A data/raw/finetune_qa_diversified 2>/dev/null)" ]]; then
    echo "[ERROR] data/raw/finetune_qa_diversified missing"
    exit 1
fi

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_finetune.py \
        --config        "$CONFIG" \
        --task          activation \
        --decoder-ckpt  "$DECODER_CKPT"

echo "[frozen_qwen3self_to_llama8b] done"
