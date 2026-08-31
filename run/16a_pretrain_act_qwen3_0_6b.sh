#!/bin/bash
#SBATCH --job-name=oracle_act_qwen3_0_6b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2                # 2× A100 80G on 1 node; 0.6B is tiny
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── Stage 1 — Qwen3-0.6B decoder, Llama-8B donor (diversified data) ─────────
#   Decoder-scaling experiment, smallest size. Validates the pipeline before
#   committing to 8B / 32B / 72B.
#
#   Donor reps : data/representations/finetune_diversified/  (Llama-3.1-8B @27)
#   Output     : checkpoints/oracle_act_qwen3_0_6b_v1-2_diverse/
#   Hardware   : 1 node × 2× A100 80G  (0.6B ~3GB bf16, easy fit)
#   Eff batch  : 192 × 2 ranks × grad_accum=2 = 768
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[oracle_act_qwen3_0_6b] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

REPS_DIR="data/representations/finetune_diversified"
[[ -d "${PROJ}/${REPS_DIR}" ]] || { echo "[ERR] missing ${REPS_DIR}"; exit 1; }

srun --unbuffered torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_train.py \
        --config experiments/training/configs/qwen3_0_6b_v1-2_diverse.yaml \
        --task   activation

echo "[oracle_act_qwen3_0_6b] done"
