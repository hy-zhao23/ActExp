#!/bin/bash
#SBATCH --job-name=cache_ft_reps
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --output=./logs/data_prep/%x_%A_%a.out
#SBATCH --error=./logs/data_prep/%x_%A_%a.err
#SBATCH --array=0-9
#SBATCH --exclude=n0046

# ────────────────────────────────────────────────────────────
# Cache last-token representations for finetune data (10 tasks, 1 GPU each)
#
# Task index → dataset:
#   0  cls  fever                    (train+test, same for both --split values)
#   1  cls  geometry_of_truth
#   2  cls  relations
#   3  cls  sst2
#   4  cls  md_gender
#   5  cls  snli
#   6  cls  ner
#   7  latentqa  control
#   8  latentqa  stimulus
#   9  latentqa  stimulus_completion
#
# Output npy naming:
#   cls:      classification_{dataset}_{split}.npy   (unchanged)
#   latentqa: latentqa_{source}_{split}.npy          (split-aware)
#
# GPU 内存 (A100-SXM4-80GB):
#   Model Llama-3.1-8B bf16        ~16 GB
#   CLS   batch=2048 × 64 tok      +16 GB  →  ~32 GB total
#   LQA   budget=48000 tokens      + 6 GB  →  ~22 GB total
#
# 提交 train（已完成，重跑改名）:
#   sbatch --array=7-9 run/04_cache_finetune_reps.sh
#
# 提交 eval latentqa（新增）:
#   sbatch --array=7-9 run/04_cache_finetune_reps.sh --split eval
#
# 单独调试:
#   sbatch --array=7 run/04_cache_finetune_reps.sh --split eval
# ────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"

SPLIT="train"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --split) SPLIT="$2"; shift 2 ;;
        *) echo "[warn] unknown arg: $1"; shift ;;
    esac
done

echo "[cache_ft_reps] task=${SLURM_ARRAY_TASK_ID}  split=${SPLIT}  node=$(hostname)"

python "${PROJ}/experiments/data_prep/cache_finetune_reps.py" \
    --task       "${SLURM_ARRAY_TASK_ID}"               \
    --model-name "meta-llama/Llama-3.1-8B-Instruct"     \
    --layer      27                                      \
    --data-dir   "${PROJ}/data/raw/finetune"             \
    --split      "${SPLIT}"                              \
    --out-dir    "${PROJ}/data/representations/finetune"

echo "[cache_ft_reps] task=${SLURM_ARRAY_TASK_ID} split=${SPLIT} done"
