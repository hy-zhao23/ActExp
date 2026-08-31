#!/bin/bash
#SBATCH --job-name=ao_eval_sk
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --qos=standard
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ────────────────────────────────────────────────────────────
# Step 2b: Secret Keeping 三项评测（Gender / Taboo / SSC）
# Run from the repository root: sbatch run/baseline_oracle/02_eval_secret_keeping.sh
# ────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"

CODE_DIR="${PROJ}/experiments/baseline/activation_oracles"
cd "$CODE_DIR"
echo "[eval_sk] 工作目录: $CODE_DIR"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHDYNAMO_DISABLE=1

echo "[eval_sk] --- Gender ---"
python experiments/gender_open_ended_eval.py

echo "[eval_sk] --- Taboo ---"
python experiments/taboo_open_ended_eval.py

echo "[eval_sk] --- SSC ---"
python experiments/ssc_open_ended_eval.py

echo "[eval_sk] ✓ Secret Keeping 评测完成"
