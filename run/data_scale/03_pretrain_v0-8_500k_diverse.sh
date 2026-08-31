#!/bin/bash
#SBATCH --job-name=ds_act_v0-8_500k_diverse
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --qos=standard                         # GPU 当前不紧张 + resume 可靠 (per-eval ckpt + early_stop_state.json)
#SBATCH --requeue
#SBATCH --time=20:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── data-scale 2/4: v1-2 setting on 500k DIVERSE recon (Stage 1) ────────────
#   500k train_split → 400k recon (recon_ratio=0.8). 与 exp 1 数据规模一致,
#   唯一变量是 source: diverse 而非 old wiki+sci.
#   Step budget: 260 steps/ep × 16 ep ≈ 4,160 steps → ~12h 实际, 20h limit.
#
# ⚠ Pre-flight checks (config 仍含 TODO 占位):
#   - data/raw/<diverse_subdir>/*.jsonl                    must exist
#   - data/representations/<diverse_subdir>/*.npy          must exist + match shards
#   - configs/data_scale/v1-2_500k_diverse.yaml            TODO 字段全部填好
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

CONFIG="experiments/training/configs/data_scale/v0-8_500k_diverse_fullqa.yaml"
if grep -q "TODO" "$CONFIG"; then
    echo "[ABORT] $CONFIG still contains TODO placeholders. Fill diverse data sources first."
    grep -n "TODO" "$CONFIG"
    exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[ds_act_v0-8_500k_diverse] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_train.py \
        --config "$CONFIG" \
        --task activation

echo "[ds_act_v0-8_500k_diverse] done"
