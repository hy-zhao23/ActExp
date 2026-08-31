#!/bin/bash
#SBATCH --job-name=oracle_act_qwen3_32b
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1                # 4 nodes × 1× A100 80G = 4 ranks
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --qos=standard
#SBATCH --time=30:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004,n0049

# ── Stage 1 — Qwen3-32B decoder, Llama-8B donor (diversified data) ──────────
#   Decoder-scaling experiment, large size. Plain DDP works because the LM is
#   frozen (bf16 ~64 GB per rank) — only the cross-attn adapter trains, so no
#   optimizer / grad state for the LM. Gradient checkpointing + small
#   per-rank batch keeps activations under control.
#
#   Donor reps : data/representations/finetune_diversified/  (Llama-3.1-8B @27)
#   Output     : checkpoints/oracle_act_qwen3_32b_v1-2_diverse/
#   Hardware   : 4 nodes × 1× A100 80G   (parallel to 4B / 8B v1-2)
#   Eff batch  : 16 × 4 ranks × grad_accum=24 = 1536  (match 4B/8B v1-2)
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[oracle_act_qwen3_32b] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

REPS_DIR="data/representations/finetune_diversified"
[[ -d "${PROJ}/${REPS_DIR}" ]] || { echo "[ERR] missing ${REPS_DIR}"; exit 1; }

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_train.py \
        --config experiments/training/configs/qwen3_32b_v1-2_diverse.yaml \
        --task   activation

echo "[oracle_act_qwen3_32b] done"
