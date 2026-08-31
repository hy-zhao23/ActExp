#!/bin/bash
#SBATCH --job-name=s2_v0-5
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

set -euo pipefail
export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

CONFIG="experiments/training/configs/data_scale/v0-5_500k_diverse_fullqa.yaml"
STAGE1_CKPT="checkpoints/data_scale/oracle_act_v0-5_500k_diverse/final"
if [[ ! -d "$STAGE1_CKPT" ]]; then echo "[ABORT] stage1 ckpt missing"; exit 3; fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[s2_v0-5] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"
unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=4 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_finetune.py \
        --config "$CONFIG" --task activation \
        --stage1-ckpt "$STAGE1_CKPT"
echo "[s2_v0-5] done"
