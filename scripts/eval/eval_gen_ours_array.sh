#!/bin/bash
#SBATCH --job-name=eval_gen_ours
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
#   sbatch --array=1-28 --gres=gpu:a100_40g:1 scripts/eval/eval_gen_ours_array.sh manifest_ours_40g.tsv
#   sbatch --array=1-4  --gres=gpu:a100:1     scripts/eval/eval_gen_ours_array.sh manifest_ours_80g.tsv
#
# Reads row $SLURM_ARRAY_TASK_ID (1-indexed, header skipped) from the manifest
# and runs eval_gen_ours.py for that ckpt.

set -euo pipefail
export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MANIFEST="${1:?usage: $0 <manifest.tsv>}"
[[ "$MANIFEST" == /* ]] || MANIFEST="${PROJ}/scripts/eval/${MANIFEST}"

# Pick row $SLURM_ARRAY_TASK_ID, skipping header (line 1)
ROW=$(awk -v idx="$SLURM_ARRAY_TASK_ID" 'NR==idx+1' "$MANIFEST")
[[ -n "$ROW" ]] || { echo "[err] no row $SLURM_ARRAY_TASK_ID in $MANIFEST"; exit 2; }

IFS=$'\t' read -r TAG FT_CKPT STAGE1_CKPT CONFIG DECODER_B BEST_VAL <<< "$ROW"

echo "[eval_gen_ours] node=${SLURMD_NODENAME}  task=${SLURM_ARRAY_TASK_ID}"
echo "[eval_gen_ours] tag=${TAG}  decoder=${DECODER_B}B  best_val=${BEST_VAL}"
echo "[eval_gen_ours] ft=${FT_CKPT}"
echo "[eval_gen_ours] stage1=${STAGE1_CKPT}"
echo "[eval_gen_ours] config=${CONFIG}"

OUT_DIR="${PROJ}/out/eval/${TAG}"
mkdir -p "${OUT_DIR}"

# Prefer the ckpt's own config.yaml (matches exact adapter arch at training time)
# over the manifest-supplied training config (which may be a renamed v2).
FT_DIR="${PROJ}/checkpoints/${FT_CKPT}"
FT_DIR_PARENT="${FT_DIR%/*}"
if [[ -f "${FT_DIR_PARENT}/config.yaml" ]]; then
    CFG_PATH="${FT_DIR_PARENT}/config.yaml"
    echo "[eval_gen_ours] using ckpt-bundled config: ${CFG_PATH}"
else
    CFG_PATH="experiments/training/configs/${CONFIG}"
    echo "[eval_gen_ours] using manifest config: ${CFG_PATH}"
fi

EXTRA_FLAGS=()
if [[ "${NO_STOP_IDS:-0}" == "1" ]]; then
    EXTRA_FLAGS+=(--no-stop-ids)
    echo "[eval_gen_ours] NO_STOP_IDS=1 → passing --no-stop-ids"
fi

python experiments/eval/eval_gen_ours.py \
    --config       "${CFG_PATH}" \
    --stage1-ckpt  "${PROJ}/checkpoints/${STAGE1_CKPT}" \
    --exact-ckpts  "${TAG}:${PROJ}/checkpoints/${FT_CKPT}" \
    --n-samples    100 \
    --output-dir   "${OUT_DIR}" \
    "${EXTRA_FLAGS[@]}"

echo "[eval_gen_ours] done  tag=${TAG}"
