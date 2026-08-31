#!/bin/bash
#SBATCH --job-name=ao_stage2_qa_ft_lr1e-4
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --time=50:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── AO stage-2 QA ft, lr=1e-4 variant ────────────────────────────────────
#   配套 v1-2_2M_lr1e-4 的 1e-4 对照实验。
#   前序 job 1002432 (lr=5e-4) 已证伪：best eval 出现在 warmup 中段 lr≈3.85e-4，
#   越过 4e-4 后 eval 单调上升，patience=6 早停。
#   当前 train.py 默认已降到 5e-5；本 job 用 AO_STAGE2_LR=1e-4 测中间档，
#   与 v1-2_2M_lr1e-4 (job 1011072) 对齐做 AO vs Oracle 对照。
#
#   独立 save_dir / run_name 与既有 v2 (lr=5e-5) checkpoint 隔离，避免误 resume。
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

REPS_DIR="${PROJ}/data/representations/finetune_qwen3_l27"
if [[ ! -f "${REPS_DIR}/metadata.json" ]]; then
    echo "[ao_stage2_qa_ft_lr1e-4] ERROR: ${REPS_DIR}/metadata.json missing. Run cache.sh first." >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

export AO_STAGE2_LR=1.0e-4                   # ← 本 job 唯一改动
export AO_STAGE2_TRAIN_BS=128
export AO_STAGE2_GRAD_ACCUM=3
export AO_STAGE2_EVAL_BS=128
export AO_STAGE2_SAVE_STEPS=250
export AO_STAGE2_EVAL_STEPS=250
export AO_STAGE2_EPOCHS=10
export AO_STAGE2_SUBSAMPLE_RATIO=0.617

# 独立 save_dir / run_name (lr1e-4 隔离)
export AO_STAGE2_SAVE_DIR="${PROJ}/checkpoints/ao_stage2_qa_ft_Qwen3-4B_lr1e-4"
export AO_STAGE2_RUN_NAME="ao_stage2_qa_ft_l27_lr1e-4"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[ao_stage2_qa_ft_lr1e-4] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}  lr=${AO_STAGE2_LR}"

unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/baseline/ao_stage2_qa_ft/train.py

echo "[ao_stage2_qa_ft_lr1e-4] done"
