#!/bin/bash
#SBATCH --job-name=reb_crosseval_val
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --time=4:00:00
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004

# R2 cross-eval：3 个 stage-2 ckpt × 3 个 val 子集的 teacher-forcing loss 矩阵。
# 输出 rebuttal/crosseval_val_results.json

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"


python rebuttal/eval_val_loss.py --out rebuttal/crosseval_val_results.json --batch-size 64

echo "[reb_crosseval_val] done"
