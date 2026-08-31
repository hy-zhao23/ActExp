#!/bin/bash
#SBATCH --job-name=qa_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --output=./logs/qa_gen/test_%j.out
#SBATCH --error=./logs/qa_gen/test_%j.err
#SBATCH --exclude=n0046

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

mkdir -p logs/qa_gen

echo "[test] node=${SLURMD_NODENAME}  gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L                                # shows MIG UUID / slice identity
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m scripts.test_vllm_40g
