#!/bin/bash
#SBATCH --job-name=ao_ft_qwen3_4b_l27_diversified
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004

# ── data-scale: AO baseline on FULL diverse QA (~1.13M) ───────────────────
#
# AO method (self-interp): donor=oracle=Qwen3-4B, layer 27 → hook layer 1.
# 与 v1-2 cross-attn 实验对比 (1023973 ds_ft_v1-2_500k_dive_fullqa) 形成方法对照:
#   - v1-2: Llama-3.1-8B donor (4096-d) → cross-attn adapter → Qwen3-4B
#   - AO:   Qwen3-4B donor   (2560-d) → steering vector hook  → Qwen3-4B
#   两边都用 full ~1.13M diverse QA train slots, 同 LoRA r=64/α=128 all-linear.
#
# Reps cache (Qwen3-4B layer 27): data/representations/finetune_diversified_qwen3_l27/
# Raw text: data/raw/finetune_diversified/
# QA bundles: data/raw/finetune_qa_diversified/
#
# Step budget (eff batch = 128 × 4 × 3 = 1536):
#   ~1.13M / 1536 ≈ 734 steps/ep × 10 ep ≈ 7,340 steps  (~22h 实际)
#
# 与既有 ao_stage2_qa_ft_lr1e-4 (1011072) 隔离的 save_dir, 避免误 resume.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

REPS_DIR="${PROJ}/data/representations/finetune_diversified_qwen3_l27"
if [[ ! -f "${REPS_DIR}/metadata.json" ]]; then
    echo "[ds_ao_ft_diverse_fullqa] ERROR: ${REPS_DIR}/metadata.json missing." >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

# ── AO trainer config (env vars consumed in experiments/baseline/ao_stage2_qa_ft/train.py) ──
export AO_STAGE2_REPS_SUBDIR="finetune_diversified_qwen3_l27"   # Qwen3 reps for diverse data
export AO_STAGE2_RAW_SUBDIR="finetune_diversified"              # diverse raw text
export AO_STAGE2_QA_SUBDIR="finetune_qa_diversified"            # diverse QA bundles
export AO_STAGE2_LR=1.0e-4                                      # match v1-2_2M_lr1e-4 / 1023973 对照
export AO_STAGE2_TRAIN_BS=128
export AO_STAGE2_GRAD_ACCUM=3
export AO_STAGE2_EVAL_BS=128
export AO_STAGE2_SAVE_STEPS=250
export AO_STAGE2_EVAL_STEPS=250
export AO_STAGE2_EPOCHS=10
# train_subsample_ratio: 不设 = full ~1.13M QA train slots
# (export AO_STAGE2_SUBSAMPLE_RATIO=null 不能用 env var, 只能 unset)
unset AO_STAGE2_SUBSAMPLE_RATIO

export AO_STAGE2_SAVE_DIR="${PROJ}/checkpoints/data_scale/ao_ft_qwen3_4b_l27_diversified"
export AO_STAGE2_RUN_NAME="ao_ft_qwen3_4b_l27_diversified"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[ds_ao_ft_diverse_fullqa] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}  lr=${AO_STAGE2_LR}"
echo "[ds_ao_ft_diverse_fullqa] reps=${AO_STAGE2_REPS_SUBDIR}  raw=${AO_STAGE2_RAW_SUBDIR}  qa=${AO_STAGE2_QA_SUBDIR}"

unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/baseline/ao_stage2_qa_ft/train.py

echo "[ds_ao_ft_diverse_fullqa] done"
