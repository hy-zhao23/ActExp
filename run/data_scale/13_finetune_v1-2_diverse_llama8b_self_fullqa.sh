#!/bin/bash
#SBATCH --job-name=ds_ft_v1-2_diverse_llama8b_self_fullqa
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --time=28:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004

# Self-donor Stage 2: Llama-3.1-8B-Instruct self + full diverse QA.
# Expects --dependency=afterok on 03_pretrain_v1-2_diverse_llama8b_self.sh.

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

CONFIG="experiments/training/configs/data_scale/v1-2_diverse_llama8b_self.yaml"
STAGE1_CKPT="checkpoints/data_scale/oracle_act_v1-2_diverse_llama8b_self/final"

if [[ ! -d "$STAGE1_CKPT" ]]; then
    echo "[ABORT] stage 1 ckpt missing: $STAGE1_CKPT"
    exit 3
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[ds_ft_v1-2_diverse_llama8b_self_fullqa] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

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

echo "[ds_ft_v1-2_diverse_llama8b_self_fullqa] done"
