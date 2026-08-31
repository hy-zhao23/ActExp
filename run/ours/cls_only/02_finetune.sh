#!/bin/bash
#SBATCH --job-name=bcls_ft
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --time=4:00:00
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=n0046

# ────────────────────────────────────────────────────────────
# Stage 2 finetune with baseline's 7 classification datasets
# Parameters match job 914751: scheme=a, lm_lr=1e-5, n_epochs=2,
#   batch=32, grad_accum=1, lora_r=8, all-linear targets
#
# 用法:
#   sbatch run/baseline_cls_test/02_finetune.sh
#   sbatch run/baseline_cls_test/02_finetune.sh --stage1-ckpt <path>
#
# Output: checkpoints/oracle_ft_baseline_cls/scheme_a/
# ────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"

CONFIG="${PROJ}/run/baseline_cls_test/baseline_cls.yaml"
STAGE1_CKPT="${PROJ}/checkpoints/oracle_act_qwen3_4b/step_2200"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage1-ckpt) STAGE1_CKPT="$2"; shift 2 ;;
        *) echo "[warn] unknown arg: $1"; shift ;;
    esac
done

cd "$PROJ"
echo "[bcls_ft] stage1_ckpt=${STAGE1_CKPT}"
echo "[bcls_ft] nodes=${SLURM_JOB_NODELIST}  node_rank=${SLURM_NODEID}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

torchrun --standalone --nproc_per_node=1 \
    experiments/baseline/ours/cls_only/train.py \
        --config      "${CONFIG}" \
        --task        activation \
        --stage1-ckpt "${STAGE1_CKPT}" \
        --scheme      a

echo "[bcls_ft] done"
