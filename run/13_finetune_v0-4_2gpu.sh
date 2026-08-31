#!/bin/bash
#SBATCH --job-name=oracle_ft_v4-0_2gpu
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1                # 1× A100 80G per node × 2 nodes = 2 total
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --qos=standard
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── Stage 2 finetune on v4-0 adapter — 2-GPU fallback ────────────────────
#   Same config as 4-node run; per-GPU batch=256 unchanged.
#   Effective batch: 256 × 2 × grad_accum=1 = 512 (half of 4-node).
#   Submit alongside 13_finetune_v4-0.sh; whichever gets scheduled first, runs.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[oracle_ft_v4-0_2gpu] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

if [[ ! -d "data/raw/finetune_qa" || -z "$(ls -A data/raw/finetune_qa 2>/dev/null)" ]]; then
    echo "[parse_qa] data/raw/finetune_qa missing — running parse"
    python -m scripts.parse_qa
fi

srun --unbuffered torchrun \
    --nnodes=2 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_finetune.py \
        --config      experiments/training/configs/qwen3_4b_v0-4.yaml \
        --task        activation \
        --stage1-ckpt checkpoints/oracle_act_qwen3_4b_v4-0/final

echo "[oracle_ft_v4-0_2gpu] done"
