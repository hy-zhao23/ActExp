#!/bin/bash
#SBATCH --job-name=ao_stage2_debug40g
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# ── DEBUG: 验证 ao_patches eval/save 是否真触发 ───────────────────────────
#   单卡 40G A100，eval_steps=50/save_steps=50 让 [eval]/[save] 早出现。
#   per-rank bs=32, grad_accum=1 → eff=32（debug，不关心训练效果）。
#   独立 save_dir 避免污染生产 ckpt；DISABLE_RESUME=1 强制从零开始。
#   wall 4h 足够触发数十次 eval/save 验证 patch。
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

REPS_DIR="${PROJ}/data/representations/finetune_qwen3_l27"
if [[ ! -f "${REPS_DIR}/metadata.json" ]]; then
    echo "[debug40g] ERROR: ${REPS_DIR}/metadata.json missing." >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

# Debug 覆写（被 train.py 的 env-var 钩子读取）
export AO_STAGE2_TRAIN_BS=32
export AO_STAGE2_EVAL_BS=32
export AO_STAGE2_GRAD_ACCUM=1
export AO_STAGE2_EVAL_STEPS=50
export AO_STAGE2_SAVE_STEPS=50
export AO_STAGE2_EPOCHS=1
export AO_STAGE2_SAVE_DIR="${PROJ}/checkpoints/ao_stage2_qa_ft_debug40g"
export AO_STAGE2_RUN_NAME="ao_stage2_debug40g_l27"
export AO_STAGE2_DISABLE_RESUME=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[debug40g] node=${SLURM_JOB_NODELIST}  master=${MASTER_ADDR}:${MASTER_PORT}"

unset SLURM_TRES_PER_TASK

srun --unbuffered torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    experiments/baseline/ao_stage2_qa_ft/train.py

echo "[debug40g] done"
