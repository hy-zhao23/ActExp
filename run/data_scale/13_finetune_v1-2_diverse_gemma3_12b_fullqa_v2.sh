#!/bin/bash
#SBATCH --job-name=ds_ft_v1-2_diverse_gemma3_12b_fullqa_v2
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004

# Gemma3-12B donor ft v2 rerun after 5/11 train_utils.py chat-template fix.
# Reuses Stage 1 act ckpt 1025530 (unaffected — Stage 1 uses flat path).

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

CONFIG="experiments/training/configs/data_scale/v1-2_diverse_gemma3_12b_v2.yaml"
STAGE1_CKPT="checkpoints/data_scale/oracle_act_v1-2_diverse_gemma3_12b/final"

if [[ ! -d "$STAGE1_CKPT" ]]; then
    echo "[ABORT] stage 1 ckpt missing: $STAGE1_CKPT"
    exit 3
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[ds_ft_v1-2_diverse_gemma3_12b_fullqa_v2] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

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

echo "[ds_ft_v1-2_diverse_gemma3_12b_fullqa_v2] done"
