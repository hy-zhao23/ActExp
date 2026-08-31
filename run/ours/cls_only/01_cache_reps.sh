#!/bin/bash
#SBATCH --job-name=bcls_cache
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --output=./logs/data_prep/%x_%j.out
#SBATCH --error=./logs/data_prep/%x_%j.err
#SBATCH --exclude=n0046

# ────────────────────────────────────────────────────────────
# Cache last-token reps for baseline's 7 training cls datasets
# Processes all datasets sequentially on 1 GPU.
#
# Output: data/representations/finetune_baseline_cls/
# ────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"

echo "[bcls_cache] node=$(hostname)"

python "${PROJ}/experiments/baseline/ours/cls_only/cache_reps.py" \
    --model-name "meta-llama/Llama-3.1-8B-Instruct" \
    --layer      27                                  \
    --out-dir    "${PROJ}/data/representations/finetune_baseline_cls"

echo "[bcls_cache] all datasets done"
