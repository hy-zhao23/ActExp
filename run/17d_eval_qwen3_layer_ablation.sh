#!/bin/bash
#SBATCH --job-name=oracle_eval_layer_abl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --time=2:00:00
#SBATCH --array=0-5
#SBATCH --output=./logs/gpu/%x_%A_%a.out
#SBATCH --error=./logs/gpu/%x_%A_%a.err
#SBATCH --exclude=n0046,n0004

# ── Open-generation eval for layer ablation (l3,l9,l15,l21,l27,l33) ───────
# Mirrors run/14_eval_gen.sh; one task per layer via SLURM array.
#
# Submit:  sbatch run/17d_eval_qwen3_layer_ablation.sh
# Output:  out/eval/oracle_ft_qwen3_4b_l${L}_diversified/{source}.jsonl + summary.txt
#
# l27 uses the original qwen3_4b_diversified_stage1.yaml (no per-layer config);
# all others use qwen3_4b_l${L}_diversified.yaml.
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

LAYERS=(3 9 15 21 27 33)
L="${LAYERS[$SLURM_ARRAY_TASK_ID]}"

if [[ "$L" == "27" ]]; then
    CONFIG="${PROJ}/experiments/training/configs/qwen3_4b_diversified_stage1.yaml"
else
    CONFIG="${PROJ}/experiments/training/configs/qwen3_4b_l${L}_diversified.yaml"
fi

STAGE1_CKPT="${PROJ}/checkpoints/oracle_act_qwen3_4b_l${L}_diversified/final"
FT_DIR="${PROJ}/checkpoints/oracle_ft_qwen3_4b_l${L}_diversified"
OUTPUT_DIR="${PROJ}/out/eval/oracle_ft_qwen3_4b_l${L}_diversified"

for p in "$CONFIG" "$STAGE1_CKPT" "$FT_DIR"; do
    if [[ ! -e "$p" ]]; then
        echo "[error] missing path: $p" >&2
        exit 1
    fi
done

N_SAMPLES="${N_SAMPLES:-50}"

echo "[eval_l${L}] node=${SLURMD_NODENAME}  array_idx=${SLURM_ARRAY_TASK_ID}"
echo "[eval_l${L}] config=${CONFIG}"
echo "[eval_l${L}] stage1=${STAGE1_CKPT}"
echo "[eval_l${L}] ft_dir=${FT_DIR}"
echo "[eval_l${L}] output=${OUTPUT_DIR}"
echo "[eval_l${L}] n_samples=${N_SAMPLES}"

python experiments/eval/eval_gen_ours.py \
    --config       "${CONFIG}" \
    --stage1-ckpt  "${STAGE1_CKPT}" \
    --ft-dirs      "ft_l${L}:${FT_DIR}" \
    --n-samples    "${N_SAMPLES}" \
    --output-dir   "${OUTPUT_DIR}"

echo "[eval_l${L}] ✓ done → ${OUTPUT_DIR}"
