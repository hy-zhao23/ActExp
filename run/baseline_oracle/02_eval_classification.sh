#!/bin/bash
#SBATCH --job-name=ao_eval_cls
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --qos=standard
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ────────────────────────────────────────────────────────────
# Step 2a: Classification 评测
# Run from the repository root: sbatch run/baseline_oracle/02_eval_classification.sh
#
# 评测 Activation Oracle 在 10+ 分类任务上的表现。
# 结果写入 experiments/activation_oracles/experiments/classification/
# ────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"

CODE_DIR="${PROJ}/experiments/baseline/activation_oracles"
cd "$CODE_DIR"
echo "[eval_cls] 工作目录: $CODE_DIR"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHDYNAMO_DISABLE=1

python experiments/classification_eval.py

echo "[eval_cls] ✓ 分类评测完成，结果见 experiments/classification/"
