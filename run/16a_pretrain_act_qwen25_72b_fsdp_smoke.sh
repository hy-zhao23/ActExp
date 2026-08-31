#!/bin/bash
#SBATCH --job-name=oracle_act_qwen25_72b_fsdp_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4                # 1 node × 4× A100 80G = 4 FSDP ranks (NVLink)
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --qos=standard
#SBATCH --time=2:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004,n0049

# ── FSDP smoke — Qwen2.5-72B, validate forward+backward before full run ─────
#   Loads 72B in bf16, FSDP-wraps decoder layers across 4 A100s on one node,
#   runs a few optimizer steps + 1 eval. If this completes, the previous
#   "CUDA illegal instruction at backward" bug is fixed and we're cleared
#   to submit the full FSDP config.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1     # single-node 4×A100: NVLink/PCIe P2P hangs on this cluster
export NCCL_IB_DISABLE=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[oracle_act_qwen25_72b_fsdp_smoke] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

REPS_DIR="data/representations/finetune_diversified"
[[ -d "${PROJ}/${REPS_DIR}" ]] || { echo "[ERR] missing ${REPS_DIR}"; exit 1; }

srun --unbuffered torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_train.py \
        --config experiments/training/configs/qwen25_72b_v1-2_diverse_fsdp_smoke.yaml \
        --task   activation

echo "[oracle_act_qwen25_72b_fsdp_smoke] done"
