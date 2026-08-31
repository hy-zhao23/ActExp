#!/bin/bash
#SBATCH --job-name=oracle_ft_layer_abl
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --time=24:00:00
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004

# ── Stage 2 (LM QA finetune) for layer ablation ───────────────────────────
# Submit:  sbatch --export=ALL,LAYER=<idx> --job-name=oracle_ft_l<idx>_div \
#                 --dependency=afterok:<stage1_jobid> \
#                 run/17c_stage2_qwen3_layer_ablation.sh
# Mirrors run/13_finetune_qwen3_l27_div.sh; config per-layer; stage1 ckpt
# auto-derived from LAYER.
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

if [[ -z "${LAYER:-}" ]]; then
    echo "[error] LAYER env var not set" >&2
    exit 1
fi

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

CONFIG="experiments/training/configs/qwen3_4b_l${LAYER}_diversified.yaml"
STAGE1_CKPT="checkpoints/oracle_act_qwen3_4b_l${LAYER}_diversified/final"

if [[ ! -f "$CONFIG" ]]; then
    echo "[error] config not found: $CONFIG" >&2
    exit 1
fi
if [[ ! -d "$STAGE1_CKPT" ]]; then
    echo "[error] stage1 ckpt missing: $STAGE1_CKPT" >&2
    ls "checkpoints/oracle_act_qwen3_4b_l${LAYER}_diversified/" 2>&1 || true
    exit 1
fi
if [[ ! -d "data/raw/finetune_qa_diversified" || -z "$(ls -A data/raw/finetune_qa_diversified 2>/dev/null)" ]]; then
    echo "[error] data/raw/finetune_qa_diversified missing" >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[stage2_l${LAYER}] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[stage2_l${LAYER}] config=${CONFIG}"
echo "[stage2_l${LAYER}] stage1_ckpt=${STAGE1_CKPT}"

unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_finetune.py \
        --config      "${CONFIG}" \
        --task        activation \
        --stage1-ckpt "${STAGE1_CKPT}"

echo "[stage2_l${LAYER}] done"
