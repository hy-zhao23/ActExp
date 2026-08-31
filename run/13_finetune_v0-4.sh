#!/bin/bash
#SBATCH --job-name=oracle_ft_v4-0
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1                # 1× A100 80G per node × 4 nodes = 4 total
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --qos=standard
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── Stage 2 finetune on v4-0 adapter (frozen adapter + LoRA) ──────────────
#   Stage 1 ckpt: checkpoints/oracle_act_qwen3_4b_v4-0/final (eval_loss=1.4052; was /best before prune refactor)
#   LoRA: r=64, α=128, dropout=0.1, target=all-linear
#   lm_lr=5e-5, min_lr_ratio=0.05, n_epochs=4
#   Hardware: 4 nodes × 1× A100 80G = 4 GPUs
#   Effective batch: 256 × 4 ranks × grad_accum=1 = 1024
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[oracle_ft_v4-0] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

# SLURM 25 bug: TRES_PER_TASK/CPUS_PER_TASK conflict — keep CPUS_PER_TASK, drop TRES.
unset SLURM_TRES_PER_TASK

# parse QA shards if not yet done (idempotent; ~2-3 min on CPU)
if [[ ! -d "data/raw/finetune_qa" || -z "$(ls -A data/raw/finetune_qa 2>/dev/null)" ]]; then
    echo "[parse_qa] data/raw/finetune_qa missing — running parse"
    python -m scripts.parse_qa
fi

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_finetune.py \
        --config      experiments/training/configs/qwen3_4b_v0-4.yaml \
        --task        activation \
        --stage1-ckpt checkpoints/oracle_act_qwen3_4b_v4-0/final

echo "[oracle_ft_v4-0] done"
