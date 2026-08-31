#!/bin/bash
#SBATCH --job-name=eval_score_scaling_rescore
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --time=1:30:00
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004

# Decoder-scaling 表重打分（8B v3 新 eval + 修复两个被 resume-bug 坍缩的旧打分）。
#
# 背景：eval_score.py 的 resume 按 (dataset, idx, slot_kind, variant) 去重，缺
# question 维度 → wiki 每个 idx 有多个 fact 问题，resume 一次 1500 行坍缩成 591 行，
# metrics_summary 就是在这 591 行(wiki 每 source 仅 25 条)上算的。
# 修法：挪走坏的 scores.jsonl 强制全新打分（生成文件本身是完整的 1500 行，没问题）。

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

for d in ours_qwen3_8b_v2_stopids ours_qwen3_0_6b_v2_stopids; do
    f="out/eval/${d}/scores.jsonl"
    if [[ -f "$f" ]] && [[ $(wc -l < "$f") -lt 1500 ]]; then
        mv "$f" "${f}.partial591_backup"
        echo "[rescore] moved partial ${f} → .partial591_backup"
    fi
done

python experiments/eval/eval_score.py \
    --eval-dir out/eval/ours_qwen3_8b_v3 \
    --eval-dir out/eval/ours_qwen3_8b_v2_stopids \
    --eval-dir out/eval/ours_qwen3_0_6b_v2_stopids \
    --bertscore-model microsoft/deberta-xlarge-mnli \
    --no-bleurt

echo "[score] ✓ done"
