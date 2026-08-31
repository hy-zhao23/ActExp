#!/bin/bash
#SBATCH --job-name=oracle_act_qwen3_32b_fsdp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4                # 1 node × 4× A100 80G = 4 FSDP ranks
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --time=20:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004,n0049

# ── FSDP sanity check — Qwen3-32B, single node 4× A100 ──────────────────────
#   Same training math as 16a_pretrain_act_qwen3_32b.sh but with FSDP wrap
#   (FULL_SHARD) replacing single-rank-per-node DDP. Output dir is suffixed
#   `_fsdp` so it doesn't clobber the already-completed 32B ckpt.
#
#   If loss curve matches the original 32B run within ~50 steps, FSDP path
#   is validated and we can apply the same wrap to Qwen2.5-72B.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[oracle_act_qwen3_32b_fsdp] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

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
        --config experiments/training/configs/qwen3_32b_v1-2_diverse_fsdp.yaml \
        --task   activation

echo "[oracle_act_qwen3_32b_fsdp] done"
