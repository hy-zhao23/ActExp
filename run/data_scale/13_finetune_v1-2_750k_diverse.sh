#!/bin/bash
#SBATCH --job-name=ds_ft_v1-2_750k_diverse
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

# ── data-scale 750k 变体: stage-2 ft only, 共享 v1-2_500k_diverse stage-1 ckpt ──
#   train_subsample_ratio=0.7889  →  ~750k QA train slots
#   Step budget: 488 steps/ep × 10 ep ≈ 4,880 steps → ~14h 实际, 24h limit.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

CONFIG="experiments/training/configs/data_scale/v1-2_750k_diverse.yaml"
STAGE1_CKPT="checkpoints/data_scale/oracle_act_v1-2_500k_diverse/final"   # 共享自 v1-2_500k_diverse

if grep -q "TODO" "$CONFIG"; then
    echo "[ABORT] $CONFIG still contains TODO placeholders."
    grep -n "TODO" "$CONFIG"
    exit 2
fi
if [[ ! -d "$STAGE1_CKPT" ]]; then
    echo "[ABORT] stage 1 ckpt missing: $STAGE1_CKPT"
    exit 3
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[ds_ft_v1-2_750k_diverse] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

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

echo "[ds_ft_v1-2_750k_diverse] done"
