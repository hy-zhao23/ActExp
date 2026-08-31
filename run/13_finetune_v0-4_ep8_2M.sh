#!/bin/bash
#SBATCH --job-name=oracle_ft_v0-4_ep8_2M
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1                # 1× A100 80G per node × 4 nodes = 4 total
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --qos=standard
#SBATCH --time=60:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── Stage 2 finetune: v0-4 with n_epochs=8 on 2M-subsample data ────────────
#   LR-floor ablation against v0-4 baseline (4ep × 3.24M, best eval=1.6205).
#   Single conceptual change: longer cosine schedule + smaller dataset
#     n_epochs:              4    → 8
#     train_subsample_ratio: 1.0  → 0.617   (per-source stratified, seed=42)
#   eff_batch (1024), peak lr, warmup, min_lr_ratio all unchanged.
#
#   Stage 1 ckpt: checkpoints/oracle_act_qwen3_4b_v4-0/final (unchanged dir name)
#   Compute: ~53h wall × 4 GPUs = 212 GPU·h, fits one 60h job (no chain).
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[oracle_ft_v0-4_ep8_2M] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

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
        --config      experiments/training/configs/qwen3_4b_v0-4_ep8_2M.yaml \
        --task        activation \
        --stage1-ckpt checkpoints/oracle_act_qwen3_4b_v4-0/final

echo "[oracle_ft_v0-4_ep8_2M] done"
