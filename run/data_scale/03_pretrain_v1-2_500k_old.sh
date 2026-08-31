#!/bin/bash
#SBATCH --job-name=ds_act_v1-2_500k_old
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --qos=standard
#SBATCH --time=20:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── data-scale 1/4: v1-2 setting on 500k OLD wiki+sci mix (Stage 1) ─────────
#   500k recon train data (250k wiki + 250k sci, n_samples=312500/source).
#   Step budget: 326 steps/ep × 16 ep ≈ 5,200 optimizer steps
#                → ~3× shorter than v1-2_ep16 (1.6M/16ep ≈ 16,700 steps).
#   Time est: ~v1-2_ep16 走 50h，500k 同 epoch ≈ 16h → 留 20h buffer.
#   See [configs/data_scale/v1-2_500k_old.yaml] header for full context.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[ds_act_v1-2_500k_old] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_train.py \
        --config experiments/training/configs/data_scale/v1-2_500k_old.yaml \
        --task activation

echo "[ds_act_v1-2_500k_old] done"
