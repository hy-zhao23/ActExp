#!/bin/bash
#SBATCH --job-name=ao_eval_cache
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --qos=standard
#SBATCH --time=1:00:00
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=n0046

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"

cd "$PROJ"
echo "[ao_eval_cache] building Llama activation cache from AO test set"

python experiments/eval/build_ao_eval_cache.py

echo "[ao_eval_cache] done"
