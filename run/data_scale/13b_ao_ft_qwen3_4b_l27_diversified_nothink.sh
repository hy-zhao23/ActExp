#!/bin/bash
#SBATCH --job-name=ao_ft_qwen3_4b_l27_diversified_nothink
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

# ── AO baseline rerun: Qwen3-4B self-interp with LatentQA-style no-think
# chat template ───────────────────────────────────────────────────────────
#
# Parent run 1026781 (ao_ft_qwen3_4b_l27_diversified) plateaued at val loss
# 4.3632 / ppl 78.51 because Qwen3's official chat template injects an empty
# `<think>\n\n</think>\n\n` block (4 fixed tokens) before the assistant
# content; those tokens stay in the loss mask and (a) dilute the reported
# loss, (b) drain 30% of the per-step gradient signal away from the actual
# answer tokens. AO_USE_NO_THINK_TEMPLATE=1 swaps in a custom Qwen3 template
# (mirroring LatentQA's DECODER_CHAT_TEMPLATES) that emits
#     <|im_start|>assistant\n<answer><|im_end|>\n
# directly, with no think block.
#
# All other hyperparameters match 1026781 exactly so val loss is comparable.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

REPS_DIR="${PROJ}/data/representations/finetune_diversified_qwen3_l27"
if [[ ! -f "${REPS_DIR}/metadata.json" ]]; then
    echo "[ao_ft_qwen3_nothink] ERROR: ${REPS_DIR}/metadata.json missing." >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

# ── AO trainer config ─────────────────────────────────────────────────────
export AO_STAGE2_REPS_SUBDIR="finetune_diversified_qwen3_l27"
export AO_STAGE2_RAW_SUBDIR="finetune_diversified"
export AO_STAGE2_QA_SUBDIR="finetune_qa_diversified"
export AO_STAGE2_LR=1.0e-4
export AO_STAGE2_TRAIN_BS=128
export AO_STAGE2_GRAD_ACCUM=3
export AO_STAGE2_EVAL_BS=128
export AO_STAGE2_SAVE_STEPS=250
export AO_STAGE2_EVAL_STEPS=250
export AO_STAGE2_EPOCHS=10
unset AO_STAGE2_SUBSAMPLE_RATIO

# ── PATCH: skip Qwen3's mandatory empty <think> block in the chat template ──
export AO_USE_NO_THINK_TEMPLATE=1

export AO_STAGE2_SAVE_DIR="${PROJ}/checkpoints/data_scale/ao_ft_qwen3_4b_l27_diversified_nothink"
export AO_STAGE2_RUN_NAME="ao_ft_qwen3_4b_l27_diversified_nothink"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[ao_ft_qwen3_nothink] nodes=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}  lr=${AO_STAGE2_LR}"
echo "[ao_ft_qwen3_nothink] reps=${AO_STAGE2_REPS_SUBDIR}  AO_USE_NO_THINK_TEMPLATE=${AO_USE_NO_THINK_TEMPLATE}"

unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/baseline/ao_stage2_qa_ft/train.py

echo "[ao_ft_qwen3_nothink] done"
