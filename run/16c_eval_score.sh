#!/bin/bash
#SBATCH --job-name=eval_score
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --qos=standard
#SBATCH --time=0:30:00
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

set -euo pipefail
export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

python experiments/eval/eval_score.py \
    --eval-dir out/eval/ours_v1-2_500k_fullqa_ablation_nosoft \
    --eval-dir out/eval/ours_v1-2_500k_fullqa_ablation_randadapter_seed0 \
    --eval-dir out/eval/ours_v1-2_500k_fullqa_ablation_randadapter_seed1 \
    --eval-dir out/eval/ours_v1-2_500k_fullqa_ablation_randadapter_seed2 \
    --eval-dir out/eval/ours_v1-2_500k_fullqa_ablation_stage1only \
    --eval-dir out/eval/ours_v1-2_500k_fullqa_ablation_baseonly \
    --eval-dir out/eval/ours_v1-2_500k_fullqa_ablation_randbase_seed0 \
    --bertscore-model microsoft/deberta-xlarge-mnli

echo "[score] ✓ done"
