#!/bin/bash
#SBATCH --job-name=eval_score_nostop
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

# Stopword-filtered RougeL/EM. BERTScore unaffected (handles semantics intrinsically).
# Outputs: scores_nostop.jsonl + metrics_summary_nostop.md   (vs gt)
#          scores_vs_input_text_nostop.jsonl + ..._nostop.md (vs input)

set -euo pipefail
export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

DIRS=(--eval-dir out/eval/ours_v1-2_2M_lr1e-4_step5200 --eval-dir out/eval/ao_qa_lr1e-4)

echo "[score-nostop] vs gt"
python experiments/eval/eval_score.py "${DIRS[@]}" \
    --bertscore-model microsoft/deberta-xlarge-mnli \
    --strip-stopwords

echo "[score-nostop] vs input_text"
python experiments/eval/eval_score.py "${DIRS[@]}" \
    --bertscore-model microsoft/deberta-xlarge-mnli \
    --target input_text \
    --strip-stopwords

echo "[score-nostop] ✓ done"
