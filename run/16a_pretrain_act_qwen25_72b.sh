#!/bin/bash
#SBATCH --job-name=oracle_act_qwen25_72b
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1                # 1 process / node owns all 4 GPUs
#SBATCH --gres=gpu:a100:4                  # 2 nodes × 4× A100 80G = 8 cards
#SBATCH --cpus-per-task=24                 # 6 per GPU × 4
#SBATCH --mem=192G                         # 48G × 4
#SBATCH --qos=standard
#SBATCH --time=15:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004,n0049,n0024,n0047

# ── Stage 1 — Qwen2.5-72B decoder, Llama-8B donor (diversified data) ────────
#   Decoder-scaling experiment, 70B+ tier. Frozen LM (~144 GiB bf16) is
#   sharded across each node's 4 A100s via HF `device_map="auto"` (naive
#   pipeline parallel); adapter trains on cuda:0 with DDP across the 2 nodes.
#
#   Donor reps : data/representations/finetune_diversified/  (Llama-3.1-8B @27)
#   Output     : checkpoints/oracle_act_qwen25_72b_v1-2_diverse/
#   Hardware   : 2 nodes × 4× A100 80G (8 cards; 2 model replicas)
#   Eff batch  : 64 × 2 ranks × grad_accum=12 = 1536  (match 4B/8B/32B)
#   Wall est.  : ~25 h for 4 epochs (naive PP throttles per-node to ~1 GPU)
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[oracle_act_qwen25_72b] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

REPS_DIR="data/representations/finetune_diversified"
[[ -d "${PROJ}/${REPS_DIR}" ]] || { echo "[ERR] missing ${REPS_DIR}"; exit 1; }

srun --unbuffered torchrun \
    --nnodes=2 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/training/oracle_train.py \
        --config experiments/training/configs/qwen25_72b_v1-2_diverse.yaml \
        --task   activation

echo "[oracle_act_qwen25_72b] done"
