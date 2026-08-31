#!/bin/bash
#SBATCH --job-name=oracle_act_qwen3_l27_div
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --qos=standard
#SBATCH --time=14:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── Stage 1 alignment: diversified mix (21 sources, ~470k) ────────────────
#   Hardware: 4 nodes × 1 A100 80G
#   Effective batch: 128 × 4 × grad_accum=3 = 1536
#   Reps: Qwen3-4B layer 27 (self-decoding setup, NOT Llama→Qwen3)
#   Source list: 17 Stage 2 sources + 4 *_extra supplements
#
#   Time estimate: 16 epochs on 470k rows ≈ 12-14 h wall clock.
#   v1-2 ep16 on 2M rows took 47h, this is 1/4 the data.
#
#   QOS=standard (low gets preempted hard; Stage 1 runs are long & DDP).
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

mkdir -p logs/gpu

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[oracle_act_div] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_train.py \
        --config experiments/training/configs/qwen3_4b_diversified_stage1.yaml \
        --task activation

echo "[oracle_act_div] done"
