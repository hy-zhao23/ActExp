#!/bin/bash
#SBATCH --job-name=oracle_act_v0-6
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --qos=standard
#SBATCH --time=40:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── Stage 1 v0-6: MLP + bottleneck=2048, n_tokens=64 ──────────────────────
#   Adapter params: ~367M  (isolates n_tokens effect vs v0-5)
#   Hardware: 80G A100, 4 nodes × 1 GPU = 4 ranks
#   Effective batch: 192 × 4 ranks × grad_accum=2 = 1536 (matches v0-4)
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[oracle_act_v0-6] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_train.py \
        --config experiments/training/configs/qwen3_4b_v0-6.yaml \
        --task activation

echo "[oracle_act_v0-6] done"
