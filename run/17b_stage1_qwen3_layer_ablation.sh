#!/bin/bash
#SBATCH --job-name=oracle_act_layer_abl
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --time=40:00:00
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004

# ── Stage 1 (activation alignment) for layer ablation ─────────────────────
# Submit:  sbatch --export=ALL,LAYER=<idx> --job-name=oracle_act_l<idx>_div \
#                 --dependency=afterok:<cache_jobid> \
#                 run/17b_stage1_qwen3_layer_ablation.sh
# Mirrors run/03_pretrain_act_v1-2.sh / qwen3_4b_diversified_stage1.yaml,
# config swapped per-layer.
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
if [[ ! -f "$CONFIG" ]]; then
    echo "[error] config not found: $CONFIG" >&2
    exit 1
fi

REPS_DIR="${PROJ}/data/representations/finetune_diversified_qwen3_l${LAYER}"
if [[ ! -f "${REPS_DIR}/metadata.json" ]]; then
    echo "[error] cache metadata missing: ${REPS_DIR}/metadata.json" >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[stage1_l${LAYER}] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[stage1_l${LAYER}] config=${CONFIG}"

unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_train.py \
        --config "${CONFIG}" \
        --task activation

echo "[stage1_l${LAYER}] done"
