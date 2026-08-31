#!/bin/bash
#SBATCH --job-name=oracle_ft_qwen3_0_6b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4                # 4× A100 80G on 1 node
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --qos=standard
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004,n0049

# ── Stage 2 — Qwen3-0.6B decoder, AO-style LoRA QA finetune ─────────────────
#   Loads Stage 1 ckpt from checkpoints/oracle_act_qwen3_0_6b_v1-2_diverse/final
#   LoRA r=64/α=128 all-linear, lm_lr=5e-5, adapter_lr=5e-4, n_epochs=4
#   Output : checkpoints/oracle_ft_qwen3_0_6b_v1-2_diverse/
#   Eff batch: 128 × 4 ranks × grad_accum=2 = 1024
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1     # single-node 4×A100: NVLink/PCIe P2P hangs on this cluster
export NCCL_IB_DISABLE=1      # single-node — no IB needed

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[oracle_ft_qwen3_0_6b] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

STAGE1_CKPT="checkpoints/oracle_act_qwen3_0_6b_v1-2_diverse/final"
REPS_DIR="data/representations/finetune_diversified"
for d in "${STAGE1_CKPT}" "${REPS_DIR}"; do
    [[ -d "${PROJ}/${d}" ]] || { echo "[ERR] missing ${d}"; exit 1; }
done

[[ -d "data/raw/finetune_qa_diversified" && -n "$(ls -A data/raw/finetune_qa_diversified 2>/dev/null)" ]] \
    || { echo "[ERR] data/raw/finetune_qa_diversified missing/empty"; exit 1; }

srun --unbuffered torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_finetune.py \
        --config      experiments/training/configs/qwen3_0_6b_v1-2_diverse.yaml \
        --task        activation \
        --stage1-ckpt "${STAGE1_CKPT}"

echo "[oracle_ft_qwen3_0_6b] done"
