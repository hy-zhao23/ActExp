#!/bin/bash
#SBATCH --job-name=ds_ft_v0-8_500k_diverse_fullqa
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004

# ── data-scale 3/3: v1-2 setting + 500k DIVERSE recon + FULL ~1.2M DIVERSE QA ──
#   train_subsample_ratio: null → 用全部 diverse QA train 数据 (估 ~1.2M).
#   Step budget: 781 steps/ep × 10 ep ≈ 7,810 steps → ~22h 实际, 30h limit.
#
#   Stage-1 ckpt 共享 v1-2_500k_diverse.yaml 训出来的 final/, 不需要再跑 stage-1.
#   建议 sbatch 顺序:
#     1) sbatch run/data_scale/03_pretrain_v1-2_500k_diverse.sh        → 拿到 stage1 ckpt
#     2) sbatch --dependency=afterok:<id1> run/data_scale/13_finetune_v1-2_500k_diverse.sh
#     3) sbatch --dependency=afterok:<id1> run/data_scale/13_finetune_v1-2_500k_diverse_fullqa.sh
#   2) 和 3) 共享 stage1, 可以并行 sbatch.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

CONFIG="experiments/training/configs/data_scale/v0-8_500k_diverse_fullqa.yaml"
STAGE1_CKPT="checkpoints/data_scale/oracle_act_v0-8_500k_diverse/final"   # 共享自 v1-2_500k_diverse

if grep -q "TODO" "$CONFIG"; then
    echo "[ABORT] $CONFIG still contains TODO placeholders."
    grep -n "TODO" "$CONFIG"
    exit 2
fi
if [[ ! -d "$STAGE1_CKPT" ]]; then
    echo "[ABORT] stage 1 ckpt missing: $STAGE1_CKPT"
    echo "        run 03_pretrain_v1-2_500k_diverse.sh first (or wait for it to finish)"
    exit 3
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[ds_ft_v0-8_500k_diverse_fullqa] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

if [[ ! -d "data/raw/finetune_qa" || -z "$(ls -A data/raw/finetune_qa 2>/dev/null)" ]]; then
    python -m scripts.parse_qa
fi

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_finetune.py \
        --config      "$CONFIG" \
        --task        activation \
        --stage1-ckpt "$STAGE1_CKPT"

echo "[ds_ft_v0-8_500k_diverse_fullqa] done"
