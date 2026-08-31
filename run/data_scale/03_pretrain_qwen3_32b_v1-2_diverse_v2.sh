#!/bin/bash
#SBATCH --job-name=oracle_act_qwen3_32b_v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --time=72:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004,n0049

# Qwen3-32B v2 act: hp aligned to 4B v1-2 main (lora_r=64 all-linear, n_epochs=16, adapter_lr=3e-4).
# FSDP FULL_SHARD on 1 node × 4× A100 80G (path validated via fsdp sanity).

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

CONFIG="experiments/training/configs/qwen3_32b_v1-2_diverse_v2.yaml"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[oracle_act_qwen3_32b_v2] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_train.py \
        --config "$CONFIG" \
        --task activation

echo "[oracle_act_qwen3_32b_v2] done"
