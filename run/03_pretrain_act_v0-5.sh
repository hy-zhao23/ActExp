#!/bin/bash
#SBATCH --job-name=oracle_act_v5
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1                # 4 nodes × 1× A100 80G = 4 ranks (4×1 easier to schedule than 2×2)
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --qos=standard
#SBATCH --time=18:00:00                  # v4-0 took ~13h on same hardware; v5 smaller adapter → slightly faster
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── Stage 1 v5: MLP + bottleneck_dim=2048, n_tokens=32 ─────────────────────
#   Hardware: 80G A100, 4 nodes × 1 GPU = 4 ranks
#   Effective batch: 256 × 4 ranks × grad_accum=2 = 2048
#   Adapter params: ~199M (v4-0: 860M, ~4× smaller)
#   Motivation: v3/v4-0 plateau at eval≈1.41 despite train=1.27 (overfit).
#               Shrink output projection by bottleneck before more tuning.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[oracle_act_v5] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_train.py \
        --config experiments/training/configs/qwen3_4b_v0-5.yaml \
        --task activation

echo "[oracle_act_v5] done"
