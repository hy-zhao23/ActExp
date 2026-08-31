#!/bin/bash
#SBATCH --job-name=oracle_ft_qwen3_l27_div
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
#SBATCH --exclude=n0046

# ── Stage 2 ft for ours-Qwen3-self (donor=Qwen3-4B layer 27, decoder=same)
#   Depends on stage 1 alignment ckpt at:
#     checkpoints/oracle_act_qwen3_4b_l27_diversified/best
#
#   Config: qwen3_4b_diversified_stage1.yaml (finetune section uses
#           reps_subdir=finetune_diversified_qwen3_l27, output_dir=
#           checkpoints/oracle_ft_qwen3_4b_l27_diversified)
#
#   Wall: ~6-12 h on 4× A100 80G (eff batch 32 × 4 × accum 2 = 256).
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

STAGE1_CKPT="checkpoints/oracle_act_qwen3_4b_l27_diversified/final"
if [[ ! -d "$STAGE1_CKPT" ]]; then
    echo "[ERROR] stage 1 ckpt missing: $STAGE1_CKPT"
    echo "  available under checkpoints/oracle_act_qwen3_4b_l27_diversified/:"
    ls checkpoints/oracle_act_qwen3_4b_l27_diversified/ 2>&1 || true
    exit 1
fi

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[oracle_ft_qwen3_l27_div] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[oracle_ft_qwen3_l27_div] stage1_ckpt=${STAGE1_CKPT}"

unset SLURM_TRES_PER_TASK

if [[ ! -d "data/raw/finetune_qa_diversified" || -z "$(ls -A data/raw/finetune_qa_diversified 2>/dev/null)" ]]; then
    echo "[ERROR] data/raw/finetune_qa_diversified missing"
    exit 1
fi

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_finetune.py \
        --config      experiments/training/configs/qwen3_4b_diversified_stage1.yaml \
        --task        activation \
        --stage1-ckpt "${STAGE1_CKPT}"

echo "[oracle_ft_qwen3_l27_div] done"
