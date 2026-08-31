#!/bin/bash
#SBATCH --job-name=ao_stage2_qa_ft_40g
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:2                # 2 nodes × 2× a100_40g = 4 ranks (NVLink intra-node)
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --time=72:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── AO method baseline (40G port, eff=1536) ────────────────────────────────
#   与 submit.sh (80G) 配置基本对齐，两处差异：
#     1) per-rank micro bs=64（40G 装不下 bs=128）
#     2) eff_batch 提升 512 → 1536（与我们 oracle Stage 1 / Stage 2 对齐，
#        baseline vs ours 比较时 batch 维度统一）
#   bs=64 × 4 ranks × ga=6 = 1536 ✓
#
#   QOS: low (40G 卡多排队短，被抢有 --requeue + ckpt resume 兜底)
#   Wall: 50h 与 80G prod 一致；eff=1536 + ep=10 实际需 ~5d，靠 requeue 接力
#         + ES patience=6 早停大概率不会真跑满 10 ep。
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

REPS_DIR="${PROJ}/data/representations/finetune_qwen3_l27"
if [[ ! -f "${REPS_DIR}/metadata.json" ]]; then
    echo "[ao_stage2_qa_ft_40g] ERROR: ${REPS_DIR}/metadata.json missing. Run cache.sh first." >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

# 40G 适配 + eff_batch 对齐 oracle 系列 (1536)
export AO_STAGE2_TRAIN_BS=64
export AO_STAGE2_GRAD_ACCUM=6
export AO_STAGE2_EVAL_BS=64
# save/eval 加密：250 opt-step ≈ 92 min，被抢最多丢 ~1.5h（vs 默认 500=3h）
export AO_STAGE2_SAVE_STEPS=250
export AO_STAGE2_EVAL_STEPS=250

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[ao_stage2_qa_ft_40g] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/baseline/ao_stage2_qa_ft/train.py

echo "[ao_stage2_qa_ft_40g] done"
