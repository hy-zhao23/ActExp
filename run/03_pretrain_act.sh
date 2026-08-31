#!/bin/bash
#SBATCH --job-name=oracle_act
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:a100:4              # 4× A100 80G
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── GPU memory budget (ACT, per rank) ────────────────────────────────
#   Static: base 8G + LoRA 0.7G = 8.7G
#   Dynamic: logits(256×208×151k) bf16+fp32 ~49G + activations → ~73G total
#   Effective batch: 256 × 2 ranks × grad_accum=2 = 1024
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

echo "[oracle_act] node=${SLURMD_NODENAME}  GPUs=${SLURM_GPUS_ON_NODE}"

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    experiments/training/oracle_train.py \
        --config experiments/training/configs/qwen3_4b_v0-1.yaml \
        --task activation

echo "[oracle_act] done"
