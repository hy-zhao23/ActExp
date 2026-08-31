#!/bin/bash
#SBATCH --job-name=data_prep_merge
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --qos=standard
#SBATCH --output=./logs/data_prep/%x_%j.out
#SBATCH --error=./logs/data_prep/%x_%j.err
#SBATCH --exclude=n0046

# merge shards → wikipedia.jsonl / scientific.jsonl，并清理 HF cache

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"

python "${PROJ}/experiments/data_prep/download_pile_subsets.py" \
    --merge \
    --world-size 20 \
    --out-dir    "${PROJ}/data/raw"

echo "[merge] done"
