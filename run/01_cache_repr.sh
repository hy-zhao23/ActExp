#!/bin/bash
#SBATCH --job-name=cache_repr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --output=./logs/data_prep/%x_%A_%a.out
#SBATCH --error=./logs/data_prep/%x_%A_%a.err
#SBATCH --array=0-39
#SBATCH --exclude=n0046

# ────────────────────────────────────────────────────────────
# Step 1: Cache layer-25 last-token residual stream (Llama-3.1-8B-Instruct)
#
# 输出：data/representations/
#   wikipedia_rank{0..39}_of_40.npy   各 ~0.59 GB  (30K × 4096, bf16-as-uint16)
#   scientific_rank{0..39}_of_40.npy  各 ~0.39 GB  (20K × 4096, bf16-as-uint16)
#   metadata.json
#   总计 ~16.4 GB
#
# GPU 内存估算 (A100-SXM4-80GB, BATCH_SIZE=8192):
#   Llama-3.1-8B bf16                  = 16 GB
#   activations peak: B×L×D×bytes      ≈ 34 GB
#   hook 立即 offload 到 CPU，不额外占显存
#   总峰值 ~50 GB (63%)
#
# 断点续传:
#   Python 脚本每隔 CHUNK_SIZE=10_000 条写 <out>.npy.progress
#   --requeue 被抢占后自动重排，重启从上次 checkpoint 继续
#
# 提交 (需先等 data_prep 完成):
#   mkdir -p logs/cache_repr
#   sbatch run/oracle/01_cache_repr.sh
# ────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"

WORLD_SIZE=40   # ← 与 --array=0-<N-1> 的 N 保持一致

echo "[cache_repr] rank ${SLURM_ARRAY_TASK_ID}/${WORLD_SIZE}  GPU: ${CUDA_VISIBLE_DEVICES}"

python "${PROJ}/experiments/data_prep/cache_representations.py" \
    --rank       "${SLURM_ARRAY_TASK_ID}" \
    --world-size "${WORLD_SIZE}"          \
    --batch-size 3500                     \
    --data-dir   "${PROJ}/data/raw"       \
    --out-dir    "${PROJ}/data/representations"

echo "[cache_repr] rank ${SLURM_ARRAY_TASK_ID} done"
