#!/bin/bash
#SBATCH --job-name=oracle_eval_gen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=standard
#SBATCH --time=1:30:00
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=n0046

# ────────────────────────────────────────────────────────────
# Step 14: Qualitative generation eval (per-dataset, 50 samples)
#
# 对 10 个 test set 各采样50条，对比：
#   stage1 — 仅 stage1 adapter（无 LoRA）
#   ft     — stage1 adapter + LoRA（最优 checkpoint 自动选取）
#
# 用法：
#   sbatch run/14_eval_gen.sh                                # 用 default FT_DIRS
#   sbatch run/14_eval_gen.sh --ft-dirs ft:checkpoints/oracle_ft_qwen3_4b_v4-0
#   sbatch run/14_eval_gen.sh --exact-ckpts ft:checkpoints/.../step_2000
#   sbatch run/14_eval_gen.sh --n-samples 100
# ────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"

CONFIG="${PROJ}/experiments/training/configs/qwen3_4b_v0-4.yaml"
STAGE1_CKPT="${PROJ}/checkpoints/oracle_act_qwen3_4b_v4-0/final"

# default: auto-select best checkpoint from metrics.jsonl
FT_DIRS=(
    "ft:${PROJ}/checkpoints/oracle_ft_qwen3_4b_v4-0"
)
EXACT_CKPTS=()
N_SAMPLES=50
N_SAMPLES_CLS=""
N_SAMPLES_LQA=""
REPR_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ft-dirs)
            FT_DIRS=(); shift
            while [[ $# -gt 0 && "$1" != --* ]]; do FT_DIRS+=("$1"); shift; done ;;
        --exact-ckpts)
            EXACT_CKPTS=(); shift
            while [[ $# -gt 0 && "$1" != --* ]]; do EXACT_CKPTS+=("$1"); shift; done ;;
        --n-samples)     N_SAMPLES="$2";     shift 2 ;;
        --n-samples-cls) N_SAMPLES_CLS="$2"; shift 2 ;;
        --n-samples-lqa) N_SAMPLES_LQA="$2"; shift 2 ;;
        --repr-dir)      REPR_DIR="$2";      shift 2 ;;
        *) echo "[warn] unknown arg: $1"; shift ;;
    esac
done

# derive output dir from ft dir / exact ckpt path
if [[ ${#FT_DIRS[@]} -gt 0 ]]; then
    FIRST_DIR="${FT_DIRS[0]#*:}"                           # strip "NAME:"
    MODEL_TAG=$(basename "${FIRST_DIR}")                   # e.g. oracle_ft_qwen3_4b_v4-0
    OUTPUT_DIR="${PROJ}/out/eval/${MODEL_TAG}"
elif [[ ${#EXACT_CKPTS[@]} -gt 0 ]]; then
    FIRST_PATH="${EXACT_CKPTS[0]#*:}"
    MODEL_TAG=$(basename "$(dirname "${FIRST_PATH}")")     # step dir's parent = output_dir
    OUTPUT_DIR="${PROJ}/out/eval/${MODEL_TAG}"
else
    OUTPUT_DIR="${PROJ}/out/eval/stage1_only"
fi

cd "$PROJ"
echo "[eval_gen] n_samples=${N_SAMPLES}  output=${OUTPUT_DIR}"

ARGS=(
    --config       "${CONFIG}"
    --stage1-ckpt  "${STAGE1_CKPT}"
    --n-samples    "${N_SAMPLES}"
    --output-dir   "${OUTPUT_DIR}"
)
[[ ${#FT_DIRS[@]}     -gt 0 ]] && ARGS+=(--ft-dirs     "${FT_DIRS[@]}")
[[ ${#EXACT_CKPTS[@]} -gt 0 ]] && ARGS+=(--exact-ckpts "${EXACT_CKPTS[@]}")
[[ -n "${N_SAMPLES_CLS}" ]]    && ARGS+=(--n-samples-cls "${N_SAMPLES_CLS}")
[[ -n "${N_SAMPLES_LQA}" ]]    && ARGS+=(--n-samples-lqa "${N_SAMPLES_LQA}")
[[ -n "${REPR_DIR}" ]]         && ARGS+=(--repr-dir "${REPR_DIR}")

python experiments/eval/oracle_eval_gen.py "${ARGS[@]}"

echo "[eval_gen] ✓ done → ${OUTPUT_DIR}"
