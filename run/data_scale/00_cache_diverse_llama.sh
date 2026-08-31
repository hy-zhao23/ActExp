#!/bin/bash
#SBATCH --job-name=cache_diverse_llama
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1            # 40G MIG 够: model 16GB + bs=1500 act ~4GB
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=standard                         # 短 job + 抢占重跑成本低 (per-shard .progress)
#SBATCH --requeue
#SBATCH --time=4:00:00
#SBATCH --output=./logs/data_prep/%x_%A_%a.out
#SBATCH --error=./logs/data_prep/%x_%A_%a.err
#SBATCH --array=0-7
#SBATCH --exclude=n0046

# ── Cache finetune_diversified through Llama-3.1-8B-Instruct layer 27 ──────
#
# Source : data/raw/finetune_diversified/*.jsonl     (21 files, 469k records)
# Output : data/representations/finetune_diversified/{source}_rank{r}_of_8.npy
# Donor  : meta-llama/Llama-3.1-8B-Instruct, layer 27, hidden_dim=4096
#
# 为什么需要这个:
#   diverse data 现有 cache 是 Qwen3-4B (2560-d), 不兼容 v1-2 baseline (Llama 4096-d donor).
#   缓存后 data scale ablation 才能保证「donor 不变, 只换数据 / scale」.
#
# Resume: per-shard .progress file 自动续 (CHUNK_SIZE=10k 粒度)
# 估算: 469k / 8 ranks ≈ 58.6k texts/rank, ~30-45min/rank on A100 40G MIG
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

mkdir -p logs/data_prep

WORLD_SIZE=8

echo "[cache_diverse_llama] rank ${SLURM_ARRAY_TASK_ID}/${WORLD_SIZE}  GPU=${CUDA_VISIBLE_DEVICES}  node=${SLURMD_NODENAME}"

python experiments/data_prep/cache_finetune_reps.py \
    --rank       "${SLURM_ARRAY_TASK_ID}" \
    --world-size "${WORLD_SIZE}"           \
    --all                                   \
    --raw-dir    "${PROJ}/data/raw/finetune_diversified" \
    --out-dir    "${PROJ}/data/representations/finetune_diversified" \
    --model-name "meta-llama/Llama-3.1-8B-Instruct" \
    --batch-size 1500

echo "[cache_diverse_llama] rank ${SLURM_ARRAY_TASK_ID} done"
