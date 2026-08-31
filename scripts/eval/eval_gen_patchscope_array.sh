#!/bin/bash
#SBATCH --job-name=eval_gen_patchscope
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --qos=low
#SBATCH --time=04:00:00
#SBATCH --exclude=n0046,n0004,n0069
#SBATCH --output=./logs/gpu/%x_%A_%a.out
#SBATCH --error=./logs/gpu/%x_%A_%a.err
#
# Patchscope-Direct = SelfIE K=1 ablation (same QA template, same inject layer).
# Reuses experiments/eval/baseline_selfie.py with --num-placeholders 1 and
# --variant-name patchscope_direct. Decoder = source (self-explain), inject
# layer L=1 mirrors SelfIE main config.
#
# Usage:
#   sbatch --array=1-2 --gres=gpu:a100_40g:1 scripts/eval/eval_gen_patchscope_array.sh
#
# Row 1: Qwen3-4B   (self-donor, l27 cache)
# Row 2: Llama-8B   (self-donor, l27 cache)

set -euo pipefail
export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MANIFEST="${PROJ}/scripts/eval/manifest_patchscope.tsv"
ROW=$(awk -v idx="$SLURM_ARRAY_TASK_ID" 'NR==idx+1' "$MANIFEST")
[[ -n "$ROW" ]] || { echo "[err] no row $SLURM_ARRAY_TASK_ID in $MANIFEST"; exit 2; }

IFS=$'\t' read -r TAG MODEL REPS_SUBDIR DONOR_LAYER INJECT_LAYER <<< "$ROW"

echo "[eval_gen_patchscope] node=${SLURMD_NODENAME}  task=${SLURM_ARRAY_TASK_ID}"
echo "[eval_gen_patchscope] tag=${TAG}  model=${MODEL}  inject_layer=${INJECT_LAYER}"

OUT_DIR="${PROJ}/out/eval/${TAG}"
mkdir -p "${OUT_DIR}"

python experiments/eval/baseline_selfie.py \
    --split        test \
    --n-samples    100 \
    --model-name   "${MODEL}" \
    --donor-layer  "${DONOR_LAYER}" \
    --reps-subdir  "${REPS_SUBDIR}" \
    --inject-layer "${INJECT_LAYER}" \
    --num-placeholders 1 \
    --variant-name "${TAG}" \
    --output-dir   "${OUT_DIR}"

echo "[eval_gen_patchscope] done  tag=${TAG}"
