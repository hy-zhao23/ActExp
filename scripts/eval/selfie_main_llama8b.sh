#!/bin/bash
#SBATCH --job-name=selfie_main_llama8b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --qos=low
#SBATCH --time=08:00:00
#SBATCH --exclude=n0046,n0004,n0069
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err

set -euo pipefail
export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[selfie_main_llama8b] node=${SLURMD_NODENAME}"

# Main test: 111 sample/source × 15 source = 1665.
# Model = meta-llama/Llama-3.1-8B-Instruct (32 layers), cached at layer 27.
# inject_layer=3 follows the SelfIE paper recommendation (k=2-3 on LLaMA-2-13B
# 40 layers); same k=3 for both Qwen-4B and Llama-8B keeps the cross-model
# comparison fair.
# Reps subdir = finetune_diversified (the unprefixed dir IS the Llama cache;
# hidden_dim=4096, layer=27 per metadata.json).
python experiments/eval/baseline_selfie.py \
    --split test --n-samples 111 \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --donor-layer 27 \
    --reps-subdir finetune_diversified \
    --inject-layer 3 --num-placeholders 5 \
    --variant-name selfie \
    --output-dir out/eval/selfie_main_llama8b

echo "[selfie_main_llama8b] done"
