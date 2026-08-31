#!/bin/bash
#SBATCH --job-name=oracle_score_layer_abl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --time=2:00:00
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004

# ── Score layer-ablation generation jsonls (l3,l9,l15,l21,l27,l33) ────────
# Mirrors run/16c_eval_score.sh. Single job; eval_score accepts
# --eval-dir repeated.
#
# Submit:  sbatch run/17e_score_qwen3_layer_ablation.sh
# Output:  out/eval/oracle_ft_qwen3_4b_l${L}_diversified/scores.jsonl +
#          metrics_summary.md (one pair per eval-dir)
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

EVAL_ARGS=()
for L in 3 9 15 21 27 33; do
    D="out/eval/oracle_ft_qwen3_4b_l${L}_diversified"
    if [[ ! -d "$D" ]]; then
        echo "[error] missing eval dir: $D" >&2
        exit 1
    fi
    EVAL_ARGS+=(--eval-dir "$D")
done

echo "[score] node=${SLURMD_NODENAME}"
for arg in "${EVAL_ARGS[@]}"; do
    [[ "$arg" != --eval-dir ]] && echo "[score]   $arg"
done

python experiments/eval/eval_score.py \
    "${EVAL_ARGS[@]}" \
    --bertscore-model microsoft/deberta-xlarge-mnli

echo "[score] ✓ done"
