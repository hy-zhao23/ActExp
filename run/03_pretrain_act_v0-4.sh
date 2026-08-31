#!/bin/bash
#SBATCH --job-name=oracle_act_v4-0
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2                # 2× A100 80G per node × 2 nodes = 4 total
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G                        # lazy-load (memmap), 16G/node is plenty
#SBATCH --qos=standard
#SBATCH --time=18:00:00                  # v3 took ~13h on same hardware
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── Stage 1 v4-0: wiki + scientific, test_ratio=0.2, 4 epochs + early-stop ────
#   Hardware: 80G A100 (40G MIG OOMs at B=128 due to full-vocab CE logits)
#   Topology: 2 nodes × 2 GPUs = 4 ranks
#   Effective batch: 192 × 4 ranks × grad_accum=2 = 1536
#   Data: wiki 960k train + sci 640k train = 1.6M samples
#   Config changes vs v3: adapter_lr 1e-3, dropout 0.1, n_epochs 4
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))     # unique port per job (avoid conflicts with parallel sbatch)
echo "[oracle_act_v4-0] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

# SLURM 25 bug: TRES_PER_TASK/CPUS_PER_TASK conflict — keep CPUS_PER_TASK, drop TRES.
unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_train.py \
        --config experiments/training/configs/qwen3_4b_v0-4.yaml \
        --task activation

echo "[oracle_act_v4-0] done"
