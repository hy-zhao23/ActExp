#!/bin/bash
#SBATCH --job-name=oracle_ft_qwen3_32b_v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --time=72:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004,n0049

# Qwen3-32B v2 ft: FSDP FULL_SHARD across 1 node × 4× A100 80G = 4 ranks.
# NVLink intra-node all-reduce ≫ IB cross-node. Matches stage 1 act topology.
# History:
#   - 1×4 ddp (1036903)  → OOM (full bf16 32B per rank).
#   - 4×2 ddp (1038356, 38443, 40955, 40962, 41002, 41036) → multi-device DDP
#     hung in NCCL key='1' init across 6 attempts.
#   - 4×1 fsdp (1041105) → wrap OK; OOM at fwd (250 MiB short, adapter Adam
#     state DDP-replicated = ~12 GiB/rank).
#   ↳ Switched adapter to FSDP too (~3 GiB/rank state) + 1×4 topology.
# eff_global = 4 × 4 × 96 = 1536. CPU peak ≈ 64 GiB shared (4 ranks on same
# node mmap same safetensors → OS page cache shared, not 4× duplicated).

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

CONFIG="experiments/training/configs/qwen3_32b_v1-2_diverse_v2.yaml"
STAGE1_CKPT="checkpoints/oracle_act_qwen3_32b_v1-2_diverse_v2/final"

if [[ ! -d "$STAGE1_CKPT" ]]; then
    echo "[ABORT] stage 1 ckpt missing: $STAGE1_CKPT"
    exit 3
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[oracle_ft_qwen3_32b_v2] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_finetune.py \
        --config      "$CONFIG" \
        --task        activation \
        --stage1-ckpt "$STAGE1_CKPT"

echo "[oracle_ft_qwen3_32b_v2] done"
