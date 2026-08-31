#!/bin/bash
#SBATCH --job-name=ds_eval_v1-2_500k_ablation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=standard
#SBATCH --time=2:30:00
#SBATCH --array=0-6
#SBATCH --output=./logs/gpu/%x_%A_%a.out
#SBATCH --error=./logs/gpu/%x_%A_%a.err
#SBATCH --exclude=n0046,n0004

# ════════════════════════════════════════════════════════════════════════════
# v1-2 (Llama-3.1-8B donor → Qwen3-4B decoder, 500k_diverse_fullqa) ablations.
#
# Reference (main): out/eval/ours_v1-2_500k_fullqa
#                   config:   experiments/training/configs/data_scale/v1-2_500k_diverse_fullqa.yaml
#                   stage1:   checkpoints/data_scale/oracle_act_v1-2_500k_diverse/final
#                   stage2:   checkpoints/data_scale/oracle_ft_v1-2_500k_diverse_fullqa/final
#
# Array cells:
#   0  no_soft           — drop ACT_TOKEN entirely; LoRA-only QA on raw question.
#   1  random_adapter#0  — re-init Q-Former adapter (seed 0); LoRA kept.
#   2  random_adapter#1  —    "                            (seed 1)
#   3  random_adapter#2  —    "                            (seed 2)
#   4  stage1_only       — stage1 Q-Former adapter only, no LoRA at all.
#   5  base_only         — floor: pure base Qwen3-4B chat, no soft tokens, no LoRA.
#   6  random_base       — random Q-Former adapter (seed 0) + base Qwen3-4B, no LoRA.
#
# n_samples mirrors the reference run (100/source).  Use the same source list.
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

CONFIG="${PROJ}/experiments/training/configs/data_scale/v1-2_500k_diverse_fullqa.yaml"
STAGE1_CKPT="${PROJ}/checkpoints/data_scale/oracle_act_v1-2_500k_diverse/final"
FT_DIR="${PROJ}/checkpoints/data_scale/oracle_ft_v1-2_500k_diverse_fullqa"

for p in "$CONFIG" "$STAGE1_CKPT" "$FT_DIR"; do
    if [[ ! -e "$p" ]]; then
        echo "[error] missing: $p" >&2
        exit 1
    fi
done

N_SAMPLES="${N_SAMPLES:-100}"
TAG="v1-2_500k_fullqa"

case "$SLURM_ARRAY_TASK_ID" in
    0)
        MODE="no_soft"
        OUTPUT_DIR="${PROJ}/out/eval/ours_${TAG}_ablation_nosoft"
        EXTRA_ARGS=( --ablation no_soft --ft-dirs "no_soft:${FT_DIR}" )
        ;;
    1|2|3)
        SEED=$((SLURM_ARRAY_TASK_ID - 1))
        MODE="random_adapter (seed=${SEED})"
        OUTPUT_DIR="${PROJ}/out/eval/ours_${TAG}_ablation_randadapter_seed${SEED}"
        EXTRA_ARGS=(
            --ablation random_adapter
            --rand-seed "${SEED}"
            --ft-dirs "rand_adapter_seed${SEED}:${FT_DIR}"
        )
        ;;
    4)
        MODE="stage1_only"
        OUTPUT_DIR="${PROJ}/out/eval/ours_${TAG}_ablation_stage1only"
        # No --ft-dirs needed; --ablation=stage1_only ignores LoRA.
        EXTRA_ARGS=( --ablation stage1_only )
        ;;
    5)
        MODE="base_only"
        OUTPUT_DIR="${PROJ}/out/eval/ours_${TAG}_ablation_baseonly"
        # No --ft-dirs needed; --ablation=base_only ignores LoRA and adapter.
        EXTRA_ARGS=( --ablation base_only )
        ;;
    6)
        MODE="random_base (seed=0)"
        OUTPUT_DIR="${PROJ}/out/eval/ours_${TAG}_ablation_randbase_seed0"
        # Random Q-Former + base model, no LoRA.
        EXTRA_ARGS=( --ablation random_base --rand-seed 0 )
        ;;
    *)
        echo "[error] unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

mkdir -p "$(dirname "${OUTPUT_DIR}")"

echo "[ablation] cell=${SLURM_ARRAY_TASK_ID}  mode=${MODE}"
echo "[ablation] config=${CONFIG}"
echo "[ablation] stage1=${STAGE1_CKPT}"
echo "[ablation] ft=${FT_DIR}"
echo "[ablation] output=${OUTPUT_DIR}"
echo "[ablation] n_samples=${N_SAMPLES}"

python experiments/eval/eval_gen_ours.py \
    --config       "${CONFIG}" \
    --stage1-ckpt  "${STAGE1_CKPT}" \
    --n-samples    "${N_SAMPLES}" \
    --output-dir   "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}"

echo "[ablation] ✓ done → ${OUTPUT_DIR}"
