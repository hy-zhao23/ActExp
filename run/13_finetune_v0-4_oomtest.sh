#!/bin/bash
#SBATCH --job-name=oracle_ft_v4-0_oomtest
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --qos=standard
#SBATCH --time=00:30:00
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# OOM smoke test: 1 GPU × batch_size=256 (per-GPU pressure identical to 4-node run).
# Runs a few training steps then exits; we only need to survive step 1's peak allocator.

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="127.0.0.1:${MASTER_PORT}" \
    experiments/training/oracle_finetune.py \
        --config      experiments/training/configs/qwen3_4b_v0-4.yaml \
        --task        activation \
        --stage1-ckpt checkpoints/oracle_act_qwen3_4b_v4-0/final

echo "[oomtest] survived"
