#!/bin/bash
#SBATCH --job-name=eval_score
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --qos=low
#SBATCH --time=06:00:00
#SBATCH --exclude=n0046,n0004
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#
# Scores every out/eval/<tag>/ dir in one pass — sharing the BERTScorer + BleurtRunner
# instances across all dirs (saves ~30s × N dirs of model load time).
#
# Usage:
#   sbatch scripts/eval/eval_score_all.sh
#   # or to limit:
#   sbatch scripts/eval/eval_score_all.sh out/eval/ours_v1-2_200k out/eval/ours_v1-2_500k

set -euo pipefail
export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

# If no args, score every subdir that has at least one *.jsonl
if [[ $# -gt 0 ]]; then
    DIRS=("$@")
else
    DIRS=()
    for d in "${PROJ}"/out/eval/*/; do
        # Skip if no per-source jsonl yet (eval-gen not done)
        if ls "${d}"*.jsonl >/dev/null 2>&1 && \
           [[ -z "$(find "${d}" -maxdepth 1 -name 'scores*.jsonl' -newer "${d}" 2>/dev/null | head -1)" ]]; then
            # Only add if has eval-gen output (any *.jsonl that isn't scores.jsonl)
            n_data=$(find "${d}" -maxdepth 1 -name '*.jsonl' ! -name 'scores*.jsonl' | wc -l)
            if [[ "$n_data" -gt 0 ]]; then
                DIRS+=("$(basename "${d%/}")")
            fi
        fi
    done
fi

echo "[eval_score] node=${SLURMD_NODENAME}  scoring ${#DIRS[@]} dirs"
ARGS=()
for d in "${DIRS[@]}"; do
    # Accept three forms: absolute path, "out/eval/<tag>", or bare "<tag>"
    if [[ "$d" == /* ]]; then
        ARGS+=(--eval-dir "$d")
    elif [[ "$d" == out/eval/* ]]; then
        ARGS+=(--eval-dir "${PROJ}/$d")
    else
        ARGS+=(--eval-dir "${PROJ}/out/eval/$d")
    fi
done

# BLEURT skipped by default — cross-setting scores are not comparable; we use
# token_f1 / rougeL / chrf / bertscore_f1 only. Pass --with-bleurt as arg to
# enable (handled via env var below).
NO_BLEURT_FLAG="--no-bleurt"
[[ "${WITH_BLEURT:-0}" == "1" ]] && NO_BLEURT_FLAG=""

python experiments/eval/eval_score.py ${NO_BLEURT_FLAG} "${ARGS[@]}"

echo "[eval_score] done"

# Aggregate immediately
python experiments/eval/aggregate_metrics.py --eval-root "${PROJ}/out/eval" --out "${PROJ}/metric.md"
echo "[aggregate] wrote ${PROJ}/metric.md"
