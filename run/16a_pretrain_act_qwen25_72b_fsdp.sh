#!/bin/bash
#SBATCH --job-name=oracle_act_qwen25_72b_fsdp
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4                # 2 nodes × 4× A100 80G = 8 FSDP ranks
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --time=72:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004,n0049,n0069,n0048

# ── Stage 1 — Qwen2.5-72B, FSDP FULL_SHARD across 8 A100 on 2 nodes ─────────
#   io.resume:true + --requeue rides through maintenance / preemption.
#   Eff batch unchanged (1536); grad_accum halved 48→24 to compensate for 2× ranks.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Fail fast on NCCL collective hang (was hanging 15h before; see 1026798 post-mortem)
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_HEARTBEAT_TIMEOUT_SEC=1800
# Inter-node NCCL must go through IB; 5/12-5/14 BROADCAST timeouts (1026767/1029746)
# trace to NCCL falling back to TCP over slow Ethernet for cross-node collectives.
# Active IB on a100 nodes: mlx5_0 port 1 (100 Gbps); mlx5_1 is Down.
export NCCL_DEBUG=INFO              # 第一次跑开 INFO，看 'NET/IB : Using [0]mlx5_0' 之类
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0:1
export NCCL_SOCKET_IFNAME=ib0       # bootstrap socket 也走 IB 接口
export NCCL_P2P_DISABLE=1           # intra-node NVLink/PCIe P2P 已知挂（见 smoke 注释）

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[oracle_act_qwen25_72b_fsdp] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

REPS_DIR="data/representations/finetune_diversified"
[[ -d "${PROJ}/${REPS_DIR}" ]] || { echo "[ERR] missing ${REPS_DIR}"; exit 1; }

srun --unbuffered torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_train.py \
        --config experiments/training/configs/qwen25_72b_v1-2_diverse_fsdp.yaml \
        --task   activation

echo "[oracle_act_qwen25_72b_fsdp] done"
