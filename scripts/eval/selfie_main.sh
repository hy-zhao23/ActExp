#!/bin/bash
#SBATCH --job-name=selfie_main_qwen3_4b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=low
#SBATCH --time=05:00:00
#SBATCH --exclude=n0046,n0004,n0069
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err

set -euo pipefail
export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[selfie_main_qwen3_4b] node=${SLURMD_NODENAME}"

# Main test: 111 sample/source × 15 source = 1665.
# inject_layer=3 follows the SelfIE paper recommendation (k=2-3 on LLaMA-2-13B
# 40 layers). Val sweep across all 36 Qwen layers showed differences are within
# noise (EM 0.067-0.081, spread <0.015), so we use the paper-locked L=3 for
# fair cross-model comparison with the Llama-8B run.
python experiments/eval/baseline_selfie.py \
    --split test --n-samples 111 \
    --model-name Qwen/Qwen3-4B \
    --donor-layer 27 \
    --reps-subdir finetune_diversified_qwen3_l27 \
    --inject-layer 3 --num-placeholders 5 \
    --variant-name selfie \
    --output-dir out/eval/selfie_main_qwen3_4b

echo "[selfie_main_qwen3_4b] done"
