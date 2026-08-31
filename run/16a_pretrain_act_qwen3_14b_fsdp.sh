#!/bin/bash
#SBATCH --job-name=oracle_act_qwen3_14b_fsdp
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1                # 4 nodes × 1× A100 80G = 4 FSDP ranks
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004

# ── Stage 1 — Qwen3-14B, FSDP FULL_SHARD across 4 A100 on 4 nodes ───────────
#   Fills the 8B → 32B gap in the decoder-size sweep. hp aligned to 8B v2.
#   Switched from 1×4 layout to 4×1 to avoid priority wait (single 4-a100
#   node currently scarce; 1-a100 slots widely available).
#   Eff batch = 64 × 4 ranks × accum 6 = 1536 (unchanged, world_size = 4).
#   Resume safe: stage 1 ckpt stores only DDP cross_attn adapter + opt state,
#   no FSDP-sharded weights → rank-agnostic. io.resume:true.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[oracle_act_qwen3_14b_fsdp] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

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
        --config experiments/training/configs/qwen3_14b_v1-2_diverse_v2.yaml \
        --task   activation

echo "[oracle_act_qwen3_14b_fsdp] done"
