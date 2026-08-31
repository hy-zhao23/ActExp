#!/bin/bash
#SBATCH --job-name=finalize_meta
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:05:00
#SBATCH --qos=standard
#SBATCH --output=./logs/data_prep/%x_%j.out
#SBATCH --error=./logs/data_prep/%x_%j.err
#SBATCH --exclude=n0046

set -euo pipefail
export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"
mkdir -p logs/data_prep

python -m scripts.finalize_diversified_metadata
