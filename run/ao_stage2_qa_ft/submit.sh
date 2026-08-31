#!/bin/bash
#SBATCH --job-name=ao_stage2_qa_ft_v2
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --time=50:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── AO method baseline on our Stage 2 QA data (Qwen3-4B self-interp) ───────
#   Step 1 dependency: cache.sh must have produced data/representations/
#     finetune_qwen3_l27/ before this job starts.
#
#   LoRA r=64, α=128, all-linear, lr=1e-4.
#   Per-rank batch 128, grad_accum=1, 4 ranks → effective 512.
#   num_epochs=5 with held-out LM-loss early stop (patience=6, min_delta=1e-4).
#   Eval over full FinetuneDataset eval split (15,709) every 500 opt steps;
#   save latest/ always, best/ when eval_loss improves.
#
#   4 nodes × 1×A100 80G DDP, multi-node torchrun. AO 无 adapter/soft-token.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

# Sanity: reps dir must exist
REPS_DIR="${PROJ}/data/representations/finetune_qwen3_l27"
if [[ ! -f "${REPS_DIR}/metadata.json" ]]; then
    echo "[ao_stage2_qa_ft] ERROR: ${REPS_DIR}/metadata.json missing. Run cache.sh first." >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

# 80G A100 × 4 ranks: BS=128 × grad_accum=3 → eff_batch=1536 (match v0-4/v1-x oracle runs)
export AO_STAGE2_TRAIN_BS=128
export AO_STAGE2_GRAD_ACCUM=3
export AO_STAGE2_EVAL_BS=128
# 250 opt-step ≈ 1.5h on 80G → 抢占最多丢 ~1h
export AO_STAGE2_SAVE_STEPS=250
export AO_STAGE2_EVAL_STEPS=250
# Match v0-4_ep8_2M: 10 epochs (early-stop kicks in earlier), 2M data subsample
export AO_STAGE2_EPOCHS=10
export AO_STAGE2_SUBSAMPLE_RATIO=0.617

# v2 (2026-04-27): fresh save_dir + run_name so we don't resume from v1
# (lr=5e-4 → 5e-5, ao_patches now adds DDP no_sync + 10-step log throttle).
# Also patience tightened: ao_patches.apply_patches(patience=...) is hard-coded
# to 6 in train.py — keeping that for now since v0-4 ft uses 3 (~150 min plateau);
# 6 here is over-tolerant but harmless if no_sync makes steps cheap.
export AO_STAGE2_SAVE_DIR="${PROJ}/checkpoints/ao_stage2_qa_ft_Qwen3-4B_v2"
export AO_STAGE2_RUN_NAME="ao_stage2_qa_ft_l27_v2"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[ao_stage2_qa_ft] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

# SLURM 25 bug: TRES_PER_TASK/CPUS_PER_TASK conflict — keep CPUS_PER_TASK, drop TRES.
unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/baseline/ao_stage2_qa_ft/train.py

echo "[ao_stage2_qa_ft] done"
