#!/bin/bash
#SBATCH --job-name=eval_gen_qwen3_8b_v3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --time=4:00:00
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004

# Open-gen eval for Qwen3-8B v3 (stage-1 early-stop fix, patience 3→10).
#   act 1042684 best eval 2.6493 (v2: 2.7257, premature stop @1400)
#   ft  1042685 best val  1.6311 (TIMEOUT @ ep5; v2 完整跑完是 1.6263, 同曲线)
# v2 的 gen-eval 在 out/eval/ours_qwen3_8b_v2_stopids (archive 同内容)，v3 写新目录。

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

CONFIG="${PROJ}/experiments/training/configs/qwen3_8b_v1-2_diverse_v3.yaml"
STAGE1_CKPT="${PROJ}/checkpoints/oracle_act_qwen3_8b_v1-2_diverse_v3/final"
FT_DIR="${PROJ}/checkpoints/oracle_ft_qwen3_8b_v1-2_diverse_v3"
OUTPUT_DIR="${PROJ}/out/eval/ours_qwen3_8b_v3"

for p in "$CONFIG" "$STAGE1_CKPT" "$FT_DIR"; do
    if [[ ! -e "$p" ]]; then
        echo "[error] missing: $p" >&2
        exit 1
    fi
done

N_SAMPLES="${N_SAMPLES:-100}"

echo "[eval_qwen3_8b_v3] node=${SLURMD_NODENAME}"
echo "[eval_qwen3_8b_v3] stage1=${STAGE1_CKPT}"
echo "[eval_qwen3_8b_v3] ft=${FT_DIR}"
echo "[eval_qwen3_8b_v3] output=${OUTPUT_DIR}"

python experiments/eval/eval_gen_ours.py \
    --config       "${CONFIG}" \
    --stage1-ckpt  "${STAGE1_CKPT}" \
    --ft-dirs      "ours_qwen3_8b_v3:${FT_DIR}" \
    --n-samples    "${N_SAMPLES}" \
    --output-dir   "${OUTPUT_DIR}"

echo "[eval_qwen3_8b_v3] ✓ done → ${OUTPUT_DIR}"
