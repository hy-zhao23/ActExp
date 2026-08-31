#!/bin/bash
#SBATCH --job-name=ao_ft_llama31_8b_l27_diversified
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --time=28:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004

# ── AO baseline: Llama-3.1-8B self-interpretation on diversified QA ───────
#
# AO method (self-interp): donor=oracle=Llama-3.1-8B-Instruct, layer 27 → hook layer 1.
# Pair to ao_ft_qwen3_4b_l27_diversified for main-table donor=Qwen vs donor=Llama
# comparison.
#
# Reps cache (Llama-3.1-8B layer 27): data/representations/finetune_diversified/
# Raw text: data/raw/finetune_diversified/
# QA bundles: data/raw/finetune_qa_diversified/  (train; val/test in _val / _test)
#
# Per-rank batch lowered to 64 (vs Qwen3-4B 128) for 8B params on 80G A100,
# grad_accum 6 → effective batch = 64 × 6 × 4 = 1536 (matches Qwen3-4B AO).
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

REPS_DIR="${PROJ}/data/representations/finetune_diversified"
if [[ ! -f "${REPS_DIR}/metadata.json" ]]; then
    echo "[ao_ft_llama] ERROR: ${REPS_DIR}/metadata.json missing." >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

# ── AO trainer config ─────────────────────────────────────────────────────
export AO_STAGE2_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export AO_STAGE2_DONOR_LAYER=27
export AO_STAGE2_REPS_SUBDIR="finetune_diversified"             # Llama reps
export AO_STAGE2_RAW_SUBDIR="finetune_diversified"
export AO_STAGE2_QA_SUBDIR="finetune_qa_diversified"
export AO_STAGE2_LR=1.0e-4
export AO_STAGE2_TRAIN_BS=64
export AO_STAGE2_GRAD_ACCUM=6
export AO_STAGE2_EVAL_BS=64
export AO_STAGE2_SAVE_STEPS=250
export AO_STAGE2_EVAL_STEPS=250
export AO_STAGE2_EPOCHS=10
unset AO_STAGE2_SUBSAMPLE_RATIO

export AO_STAGE2_SAVE_DIR="${PROJ}/checkpoints/data_scale/ao_ft_llama31_8b_l27_diversified"
export AO_STAGE2_RUN_NAME="ao_ft_llama31_8b_l27_diversified"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[ao_ft_llama] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}  lr=${AO_STAGE2_LR}"
echo "[ao_ft_llama] model=${AO_STAGE2_MODEL_NAME}  layer=${AO_STAGE2_DONOR_LAYER}  reps=${AO_STAGE2_REPS_SUBDIR}"

unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/baseline/ao_stage2_qa_ft/train.py

echo "[ao_ft_llama] done"
