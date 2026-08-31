#!/bin/bash
#SBATCH --job-name=reb_crosseval_mix5
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --time=6:00:00
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004

# R2 收尾：mix 500k 全部 5 个 seed 过统一 cross-eval（nonwiki/wiki/full 三张考卷），
# 把 2×2 矩阵补成 3×2（mix 行报 mean±std over 5 seeds）。
# 输出独立 json，不覆盖 1096323 的 crosseval_val_results.json。

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

CFG="experiments/training/configs/rebuttal/qwen3_4b_instr_self_500k.yaml"
CK="checkpoints/rebuttal/oracle_ft_qwen3_instr_self_500k"

python rebuttal/eval_val_loss.py \
    --out rebuttal/crosseval_val_mix5seeds.json \
    --ckpt "mix_s42:${CK}/final:${CFG}" \
    --ckpt "mix_s43:${CK}_seed43/final:${CFG}" \
    --ckpt "mix_s44:${CK}_seed44/final:${CFG}" \
    --ckpt "mix_s45:${CK}_seed45/final:${CFG}" \
    --ckpt "mix_s46:${CK}_seed46/final:${CFG}"

echo "[crosseval_mix5] ✓ done → rebuttal/crosseval_val_mix5seeds.json"
