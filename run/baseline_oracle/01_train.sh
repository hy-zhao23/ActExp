#!/bin/bash
#SBATCH --job-name=ao_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100_40g:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G                   # 4 rank × ~20 GB：8G 模型中转 + dataset shard
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ────────────────────────────────────────────────────────────
# Step 1: 训练 Activation Oracle（Qwen3-4B LoRA，4×20GB MIG）
#
# ── GPU 显存（CUDA）──────────────────────────────────────────
#   Qwen3-4B LoRA（r=64）+ gradient checkpointing 每 rank 峰值：
#     模型权重 8G + AdamW 状态 1G + CUDA 开销 1G = 10 GB
#     梯度检查点重算（最长序列 L≈2000）≈ 6 GB
#     合计峰值 ≈ 16 GB → 20GB MIG（80% 利用率）✓
#
# ── CPU 内存（System RAM）────────────────────────────────────
#   4 进程各自独立加载，互不共享：
#     模型加载时 CPU↔GPU 中转  ≈ 8 GB × 4 = 32 GB
#     全量 dataset 加载再 shard ≈ 20 GB × 4 = 80 GB
#     Python / PyTorch 开销     ≈ 2 GB × 4 = 8 GB
#     合计 ≈ 120 GB → --mem=160G（留余量）✓
#
# 切换到 40GB MIG（更快训练）：
#   将 gres 改为 gpu:a100_40g:4，sft.py 里 train_batch_size=256（64/rank）
#
# 切换回全卡（最大模型，如 32B/70B）：
#   将 gres 改为 gpu:a100:4，调整 batch size 和量化配置
# ────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"

CODE_DIR="${PROJ}/experiments/baseline/activation_oracles"
cd "$CODE_DIR"
echo "[train] 工作目录: $CODE_DIR"
echo "[train] 节点: $SLURMD_NODENAME  GPUs: $SLURM_GPUS_ON_NODE"

# ── 显存分配优化 ───────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

# ── torchrun 分布式训练（4 GPU DDP） ───────────────────────
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    nl_probes/sft.py

echo "[train] ✓ 训练完成，checkpoint 保存于 ${CODE_DIR}/checkpoints/"
