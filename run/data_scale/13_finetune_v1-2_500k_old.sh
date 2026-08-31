#!/bin/bash
#SBATCH --job-name=ds_ft_v1-2_500k_old
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1                # 1× A100 80G per node × 4 nodes = 4 total
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G                        # 12G→16G: QA bundles 全部 dict 进 RAM (~2GB) + Qwen3-4B 加载临时峰值留余量
#SBATCH --qos=standard
#SBATCH --time=20:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── data-scale 1/4: v1-2 setting on 500k OLD QA (Stage 2) ───────────────────
#   3.24M QA → train_subsample_ratio=0.154 → ~500k QA samples train.
#   Stage-1 ckpt: checkpoints/data_scale/oracle_act_v1-2_500k_old/final
#   Step budget: 326 steps/ep × 10 ep ≈ 3,260 optimizer steps.
#   Time est: ~v1-2_2M_lr1e-4 (~3.24M @ 0.617 = 2M) 跑 60h，500k ≈ 1/4 → 15h.
#   留 20h buffer (lr/early-stop 行为可能不同).
#   See [configs/data_scale/v1-2_500k_old.yaml] header for full context.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[ds_ft_v1-2_500k_old] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

if [[ ! -d "data/raw/finetune_qa_old" || -z "$(ls -A data/raw/finetune_qa_old 2>/dev/null)" ]]; then
    echo "[parse_qa] data/raw/finetune_qa_old missing — running parse"
    python -m scripts.parse_qa     # default writes to data/raw/finetune_qa_old/
fi

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_finetune.py \
        --config      experiments/training/configs/data_scale/v1-2_500k_old.yaml \
        --task        activation \
        --stage1-ckpt checkpoints/data_scale/oracle_act_v1-2_500k_old/final

echo "[ds_ft_v1-2_500k_old] done"
