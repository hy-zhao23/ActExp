#!/bin/bash
#SBATCH --job-name=ds_ft_v1-2_500k_diverse_nostage1
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --time=20:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── Stage-1 ablation @ 500k QA: skip recon pretrain, joint adapter+LoRA ft ──
#
# 对比 1023972 (oracle_ft_v1-2_500k_diverse, 有 stage-1 recon) → 等数据量下 stage-1 增益.
# 也对比 1024038 (diverse_fullqa_nostage1, full 1.13M)             → 数据量影响 (no stage-1 设定下).
#
# 不传 --stage1-ckpt → adapter 从 scratch 随机初始化, 与 LoRA 一起训练.
# train_subsample_ratio=0.4437 → 500k QA, 与 1023972 完全同分布同 seed.
#
# Step budget: ~500k / 1536 ≈ 326 steps/ep × 10 ep ≈ 3,260 steps  (~10h)
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

CONFIG="experiments/training/configs/data_scale/v1-2_500k_diverse_nostage1.yaml"

if grep -q "TODO" "$CONFIG"; then
    echo "[ABORT] $CONFIG still contains TODO placeholders."
    grep -n "TODO" "$CONFIG"
    exit 2
fi

# 数据 reps 必需 (Llama-3.1-8B donor cache)
REPS_DIR="${PROJ}/data/representations/finetune_diversified"
if [[ ! -f "${REPS_DIR}/metadata.json" ]]; then
    echo "[ABORT] ${REPS_DIR}/metadata.json missing." >&2
    exit 3
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[ds_ft_v1-2_500k_diverse_nostage1] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[ds_ft_v1-2_500k_diverse_nostage1] adapter from scratch (no stage 1 ckpt), 500k subsample"

unset SLURM_TRES_PER_TASK

# 注意: 不传 --stage1-ckpt → adapter 随机初始化
srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_finetune.py \
        --config "$CONFIG" \
        --task   activation

echo "[ds_ft_v1-2_500k_diverse_nostage1] done"
