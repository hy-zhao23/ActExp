#!/bin/bash
#SBATCH --job-name=inspect_llama3_tok
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --qos=standard
#SBATCH --time=00:05:00
#SBATCH --exclude=n0046
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err

set -euo pipefail
export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

python "${PROJ}/scripts/inspect_llama3_tokens.py"
