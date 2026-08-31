#!/bin/bash
#SBATCH --job-name=ds_ft_v1-2_diverse_fullqa_nostage1
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --time=30:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── Stage-1 ablation: skip recon pretrain, joint adapter+LoRA ft on full diverse QA ──
#
# 对比 1023973 (v1-2 fullqa, 有 stage-1 recon 预训).
# 唯一变量: 是否做 stage-1 → 测 stage-1 是否必要.
#
# 不传 --stage1-ckpt → adapter 从 scratch 随机初始化, 与 LoRA 一起训练.
#
# Step budget: ~1.13M / 1536 ≈ 734 steps/ep × 10 ep ≈ 7,340 steps  (~22h)
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

CONFIG="experiments/training/configs/data_scale/v1-2_diverse_fullqa_nostage1.yaml"

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
echo "[ds_ft_v1-2_diverse_fullqa_nostage1] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[ds_ft_v1-2_diverse_fullqa_nostage1] adapter from scratch (no stage 1 ckpt)"

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

echo "[ds_ft_v1-2_diverse_fullqa_nostage1] done"
