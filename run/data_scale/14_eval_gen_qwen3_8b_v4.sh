#!/bin/bash
#SBATCH --job-name=eval_gen_qwen3_8b_v4
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

# Open-gen eval for Qwen3-8B v4（Setting C: act adapter_lr 1e-4）。

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

CONFIG="${PROJ}/experiments/training/configs/qwen3_8b_v1-2_diverse_v4_adlr1e-4.yaml"
STAGE1_CKPT="${PROJ}/checkpoints/oracle_act_qwen3_8b_v1-2_diverse_v4_adlr1e-4/final"
FT_DIR="${PROJ}/checkpoints/oracle_ft_qwen3_8b_v1-2_diverse_v4_adlr1e-4"
OUTPUT_DIR="${PROJ}/out/eval/ours_qwen3_8b_v4_adlr1e-4"

for p in "$CONFIG" "$STAGE1_CKPT" "$FT_DIR"; do
    if [[ ! -e "$p" ]]; then
        echo "[error] missing: $p" >&2
        exit 1
    fi
done

N_SAMPLES="${N_SAMPLES:-100}"

echo "[eval_qwen3_8b_v4] node=${SLURMD_NODENAME}  output=${OUTPUT_DIR}"

python experiments/eval/eval_gen_ours.py \
    --config       "${CONFIG}" \
    --stage1-ckpt  "${STAGE1_CKPT}" \
    --ft-dirs      "ours_qwen3_8b_v4_adlr1e-4:${FT_DIR}" \
    --n-samples    "${N_SAMPLES}" \
    --output-dir   "${OUTPUT_DIR}"

echo "[eval_qwen3_8b_v4] ✓ done → ${OUTPUT_DIR}"
