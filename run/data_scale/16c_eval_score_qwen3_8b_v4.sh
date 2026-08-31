#!/bin/bash
#SBATCH --job-name=eval_score_qwen3_8b_v4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --time=1:00:00
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004

set -euo pipefail
export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

python experiments/eval/eval_score.py \
    --eval-dir out/eval/ours_qwen3_8b_v4_adlr1e-4 \
    --bertscore-model microsoft/deberta-xlarge-mnli \
    --no-bleurt

echo "[score] ✓ done"
