#!/bin/bash
#SBATCH --job-name=oracle_ft_v1-2_2M_lr1e-4
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1                # 1× A100 80G per node × 4 nodes = 4 total
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --qos=standard
#SBATCH --time=60:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── Stage 2 finetune: v1-2/ep16 (q-former) + 2M data, lm_lr=1e-4 variant ──
#   原版 (job 1010956) 用 lm_lr=5e-4。前序 ao_stage2_qa_ft (job 1002432) 显示
#   5e-4 已经过冲 (best eval 出现在 warmup 中段 lr≈3.85e-4)，本 job 降到 1e-4。
#   其他超参与 v1-2_2M 完全一致以便对照。
#   Output dir: checkpoints/oracle_ft_qwen3_4b_v1-2_2M_lr1e-4 (与原 job 隔离)
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[oracle_ft_v1-2_2M_lr1e-4] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

if [[ ! -d "data/raw/finetune_qa" || -z "$(ls -A data/raw/finetune_qa 2>/dev/null)" ]]; then
    echo "[parse_qa] data/raw/finetune_qa missing — running parse"
    python -m scripts.parse_qa
fi

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_finetune.py \
        --config      experiments/training/configs/qwen3_4b_v1-2_2M_lr1e-4.yaml \
        --task        activation \
        --stage1-ckpt checkpoints/oracle_act_qwen3_4b_v1-2/ep16/final

echo "[oracle_ft_v1-2_2M_lr1e-4] done"
