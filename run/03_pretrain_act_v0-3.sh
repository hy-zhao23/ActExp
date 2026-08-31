#!/bin/bash
#SBATCH --job-name=oracle_act_v3
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2                # 2× A100 80G per node × 2 nodes = 4 total
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                        # 2 GPUs per task, lazy-load dataset
#SBATCH --qos=standard
#SBATCH --time=18:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── Stage 1 v3: wiki + scientific, test_ratio=0.2, 5 epochs + early-stop ────
#   Topology: 2 nodes × 2 GPUs (4×1 not schedulable — no 4 independent free slots)
#   Effective batch: 192 × 4 ranks × grad_accum=2 = 1536
#   Data: wiki 960k train + sci 640k train = 1.6M samples
#   Expected runtime: ~5 hours (with early-stop likely ~3-4h)
#   Resume: automatic (io.resume=true, early_stop_state.json persists)
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
echo "[oracle_act_v3] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

# SLURM 25 bug: TRES_PER_TASK/CPUS_PER_TASK conflict — keep CPUS_PER_TASK, drop TRES.
unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_train.py \
        --config experiments/training/configs/qwen3_4b_v0-3.yaml \
        --task activation

echo "[oracle_act_v3] done"
