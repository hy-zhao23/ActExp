#!/bin/bash
#SBATCH --job-name=eval_gen_ao
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --qos=standard
#SBATCH --time=1:00:00
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# Standalone AO gen — runs in parallel with the ours step of 16_eval_smoke.sh.
# 16_eval_smoke.sh will redundantly re-run AO after its ours step finishes;
# whichever writes last wins (content is deterministic, so no semantic issue).

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

OUT_AO="${PROJ}/out/eval/ao_qa_lr1e-4"

echo "[ao] gen → ${OUT_AO}"
python experiments/eval/eval_gen_ao.py \
    --ao-dirs    ao_lr1e-4:${PROJ}/checkpoints/ao_stage2_qa_ft_Qwen3-4B_lr1e-4 \
    --n-samples  50 \
    --output-dir "${OUT_AO}"

echo "[ao] ✓ done"
