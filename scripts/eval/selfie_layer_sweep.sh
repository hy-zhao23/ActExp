#!/bin/bash
#SBATCH --job-name=selfie_sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --exclude=n0046
#SBATCH --array=0-35
#SBATCH --output=./logs/gpu/%x_%A_%a.out
#SBATCH --error=./logs/gpu/%x_%A_%a.err

set -euo pipefail
export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

L=${SLURM_ARRAY_TASK_ID}
echo "[selfie_sweep L=${L}] node=${SLURMD_NODENAME}"

# 30 sample × 9 source × 1 layer = 270 inferences ≈ 30 min per array task.
# Cluster decides parallelism (currently 3 idle GPUs → ~12 rounds × 30min = ~6h wall).
python experiments/eval/baseline_selfie.py \
    --split val \
    --n-samples 30 \
    --inject-layer "${L}" --num-placeholders 5 \
    --variant-name "selfie_L${L}" \
    --output-dir "out/eval/selfie_layer_sweep/L${L}"

echo "[selfie_sweep L=${L}] done"
