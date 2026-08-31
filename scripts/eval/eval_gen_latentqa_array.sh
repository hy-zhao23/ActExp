#!/bin/bash
#SBATCH --job-name=eval_gen_latentqa
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --qos=low
#SBATCH --time=04:00:00
#SBATCH --exclude=n0046,n0004
#SBATCH --output=./logs/gpu/%x_%A_%a.out
#SBATCH --error=./logs/gpu/%x_%A_%a.err
#
# Usage:
#   sbatch --array=1-2 --gres=gpu:a100_40g:1 scripts/eval/eval_gen_latentqa_array.sh
#
# Row 1: Qwen3-4B-Instruct-2507 decoder
# Row 2: Llama-3.1-8B-Instruct  decoder (needs ~17 GB just for the LM in bf16,
#        fits comfortably on a100_40g with bs=1 generation + use_cache=False).

set -euo pipefail
export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MANIFEST="${PROJ}/scripts/eval/manifest_latentqa.tsv"
ROW=$(awk -v idx="$SLURM_ARRAY_TASK_ID" 'NR==idx+1' "$MANIFEST")
[[ -n "$ROW" ]] || { echo "[err] no row $SLURM_ARRAY_TASK_ID in $MANIFEST"; exit 2; }

IFS=$'\t' read -r TAG CKPT_DIR <<< "$ROW"

echo "[eval_gen_latentqa] node=${SLURMD_NODENAME}  task=${SLURM_ARRAY_TASK_ID}"
echo "[eval_gen_latentqa] tag=${TAG}"
echo "[eval_gen_latentqa] ckpt=${CKPT_DIR}"

OUT_DIR="${PROJ}/out/eval/${TAG}"
mkdir -p "${OUT_DIR}"

python experiments/eval/baseline_latentqa.py \
    --variants    "${TAG}:${PROJ}/${CKPT_DIR}" \
    --n-samples   100 \
    --output-dir  "${OUT_DIR}"

echo "[eval_gen_latentqa] done  tag=${TAG}"
