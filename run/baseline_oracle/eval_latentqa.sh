#!/bin/bash
#SBATCH --job-name=ao_eval_lqa
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --qos=standard
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ────────────────────────────────────────────────────────────
# Baseline AO eval on latentqa holdout (eval split)
#
# 使用 baseline 的 finetuned verbalizer (Qwen3-4B) 在
# latentqa eval split 上做 open-ended 生成评测。
#
# Run from the repository root: sbatch run/baseline_oracle/eval_latentqa.sh
# ────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"

CODE_DIR="${PROJ}/experiments/baseline/activation_oracles"
cd "$CODE_DIR"
echo "[eval_lqa] working dir: $CODE_DIR"

python experiments/latentqa_open_ended_eval.py

echo "[eval_lqa] ✓ done"
