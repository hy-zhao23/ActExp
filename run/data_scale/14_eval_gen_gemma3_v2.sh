#!/bin/bash
#SBATCH --job-name=eval_gen_gemma3_v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --time=2:00:00
#SBATCH --array=0-1
#SBATCH --output=./logs/gpu/%x_%A_%a.out
#SBATCH --error=./logs/gpu/%x_%A_%a.err
#SBATCH --exclude=n0046,n0004

# Open-gen eval for Gemma3 donor v2 ft (post 5/11 chat-template fix).
#   Array idx 0 → Gemma3-4B-IT  donor (ft 1035493, val 1.6775)
#   Array idx 1 → Gemma3-12B-IT donor (ft 1035494, val 1.6594)
# Writes to out/eval/ours_donor_gemma3_{4b,12b}_v2 (kept separate from pre-fix v1 eval).

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

SIZES=(4b 12b)
SIZE="${SIZES[$SLURM_ARRAY_TASK_ID]}"

CONFIG="${PROJ}/experiments/training/configs/data_scale/v1-2_diverse_gemma3_${SIZE}_v2.yaml"
STAGE1_CKPT="${PROJ}/checkpoints/data_scale/oracle_act_v1-2_diverse_gemma3_${SIZE}/final"
FT_DIR="${PROJ}/checkpoints/data_scale/oracle_ft_v1-2_diverse_gemma3_${SIZE}_fullqa_v2"
OUTPUT_DIR="${PROJ}/out/eval/ours_donor_gemma3_${SIZE}_v2"

for p in "$CONFIG" "$STAGE1_CKPT" "$FT_DIR"; do
    if [[ ! -e "$p" ]]; then
        echo "[error] missing: $p" >&2
        exit 1
    fi
done

N_SAMPLES="${N_SAMPLES:-50}"

echo "[eval_gemma3_${SIZE}] node=${SLURMD_NODENAME}"
echo "[eval_gemma3_${SIZE}] config=${CONFIG}"
echo "[eval_gemma3_${SIZE}] stage1=${STAGE1_CKPT}"
echo "[eval_gemma3_${SIZE}] ft=${FT_DIR}"
echo "[eval_gemma3_${SIZE}] output=${OUTPUT_DIR}"

python experiments/eval/eval_gen_ours.py \
    --config       "${CONFIG}" \
    --stage1-ckpt  "${STAGE1_CKPT}" \
    --ft-dirs      "ft_v2:${FT_DIR}" \
    --n-samples    "${N_SAMPLES}" \
    --output-dir   "${OUTPUT_DIR}"

echo "[eval_gemma3_${SIZE}] ✓ done → ${OUTPUT_DIR}"
