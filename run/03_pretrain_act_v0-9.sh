#!/bin/bash
#SBATCH --job-name=oracle_act_v0-9
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1                # 4 nodes × 1× A100 80G = 4 ranks (4×1 易调度)
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --qos=standard
#SBATCH --time=50:00:00                  # ep16 ≈ 48h；resume:true 兜底超时续训
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── Stage 1 v0-9: MLP baseline @ eff batch 1536, lr=3e-4, ep=16 + ES ─────
#   v0-3 lr=3e-4 在 step 3200 (≈ep3) ES 出 loss 1.4099；v0-4 改 1e-3 加速
#   反而可能过拟。本版回到 3e-4 (= q-former v1-2 同 lr) + ep 16 + patience 6，
#   看 MLP 在充分时间下能下探到什么 plateau。eff batch 保持 1536 不变。
#   拓扑：4 nodes × 1 GPU（与 v0-5/6/7/8 一致，比 2×2 易调度）。
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[oracle_act_v0-9] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_train.py \
        --config experiments/training/configs/qwen3_4b_v0-9.yaml \
        --task activation

echo "[oracle_act_v0-9] done"
