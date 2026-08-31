#!/bin/bash
#SBATCH --job-name=eval_score_vs_input
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

# Sanity check: how much does prediction look like the input_text (the source the
# activation came from)? High score → model is paraphrasing input, not answering Q.

set -euo pipefail
export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

python experiments/eval/eval_score.py \
    --eval-dir out/eval/ours_v1-2_2M_lr1e-4_step5200 \
    --eval-dir out/eval/ao_qa_lr1e-4 \
    --bertscore-model microsoft/deberta-xlarge-mnli \
    --target input_text

echo "[score-vs-input] ✓ done"
