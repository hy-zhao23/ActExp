#!/bin/bash
#SBATCH --job-name=reb_score_r2_g2f
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --time=2:00:00
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004

set -euo pipefail
export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"
python experiments/eval/eval_score.py \
    --eval-dir out/eval/rebuttal_r2_qatypes_g2f \
    --bertscore-model microsoft/deberta-xlarge-mnli \
    --no-bleurt
echo "[score] ✓ done"
