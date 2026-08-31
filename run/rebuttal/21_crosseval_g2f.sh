#!/bin/bash
#SBATCH --job-name=reb_crosseval_g2f
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

python rebuttal/eval_val_loss.py \
    --out rebuttal/crosseval_val_g2f.json --batch-size 64 \
    --ckpt "g2f:checkpoints/rebuttal/oracle_ft_qwen3_instr_self_gistcls_then_fact/final:experiments/training/configs/rebuttal/qwen3_4b_instr_self_gistcls_then_fact.yaml"

echo "[reb_crosseval_g2f] done"
