#!/bin/bash
#SBATCH --job-name=data_prep
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --output=./logs/data_prep/%x_%A_%a.out
#SBATCH --error=./logs/data_prep/%x_%A_%a.err
#SBATCH --array=0-19
#SBATCH --exclude=n0046

# ────────────────────────────────────────────────────────────
# Step 0: 预处理语料（20 路并行 shard）
#
# 输出：
#   data/raw/wikipedia.jsonl   1.2M 条（Wikipedia 首段，≤64 token）
#   data/raw/scientific.jsonl  0.8M 条（peS2o abstract，≤64 token）
#
# shard 数量说明：
#   peS2o v2 有 20 个 json.gz 文件，world_size=20 时每个 task 恰好分到
#   1 个文件，读完 40K 条即停；Wikipedia 有 41 个 parquet，每 task ~2 个。
#
# 提交：
#   Run from the repository root.
#   mkdir -p logs/data_prep
#   JID=$(sbatch --parsable run/oracle/00_data_prep.sh)
#   sbatch --dependency=afterok:$JID run/oracle/00_data_prep_merge.sh
# ────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"

WORLD_SIZE=20   # ← 与 --array=0-<N-1> 的 N 保持一致

python "${PROJ}/experiments/data_prep/download_pile_subsets.py" \
    --rank          "${SLURM_ARRAY_TASK_ID}" \
    --world-size    "${WORLD_SIZE}" \
    --out-dir       "${PROJ}/data/raw"

echo "[data_prep] rank ${SLURM_ARRAY_TASK_ID} done"
