#!/bin/bash
#SBATCH --job-name=ao_eval_pqa
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --qos=standard
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ────────────────────────────────────────────────────────────
# Step 2c: PersonaQA 评测（开放式 + 知识 + Yes/No）
# Run from the repository root: sbatch run/baseline_oracle/02_eval_personaqa.sh
# ────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"

CODE_DIR="${PROJ}/experiments/activation_oracles"
cd "$CODE_DIR"
echo "[eval_pqa] 工作目录: $CODE_DIR"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHDYNAMO_DISABLE=1

echo "[eval_pqa] --- PersonaQA Open-ended ---"
python experiments/personaqa_open_ended_eval.py

echo "[eval_pqa] --- PersonaQA Yes/No ---"
python experiments/personaqa_yes_no_eval.py

echo "[eval_pqa] --- PersonaQA Knowledge ---"
python experiments/personaqa_knowledge_eval.py

echo "[eval_pqa] --- PersonaQA Knowledge Yes/No ---"
python experiments/personaqa_knowledge_yes_no_eval.py

echo "[eval_pqa] ✓ PersonaQA 评测完成"
