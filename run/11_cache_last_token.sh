#!/bin/bash
#SBATCH --job-name=ao_cache_lt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --qos=standard
#SBATCH --output=./logs/gpu/%x_%A_%a.out
#SBATCH --error=./logs/gpu/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --exclude=n0046

# ────────────────────────────────────────────────────────────
# Stage 2 Step 0: 预缓存激活向量 — last-token setting
#
# 与 00_cache.sh (baseline) 的区别：
#   - Classification : 只有 single-token 变体，end_offset 固定 -1（严格 last token）
#   - LatentQA       : position_types=["window"]，window=1，end_offset 固定 -1
#   - PastLens       : 只有 single 变体（max_k_activations=1，天然 last token）
#
# 生成的 .pt 文件 config hash 不同，与 baseline 缓存共存，互不覆盖。
#
# 提交：
#   Run from the repository root: sbatch run/11_cache_last_token.sh
# ────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"

CODE_DIR="${PROJ}/experiments/data_prep"
cd "$CODE_DIR"

MODEL_NAME="Qwen/Qwen3-4B"
WORLD_SIZE=4
BATCH_SIZE=256
LATENTQA_BATCH_SIZE=64

echo "[cache_lt] Task ${SLURM_ARRAY_TASK_ID}/${WORLD_SIZE}  GPU: ${CUDA_VISIBLE_DEVICES}"
echo "[cache_lt] 模型: ${MODEL_NAME}"

# ── Step A: 序列化 HF 模型下载 ────────────────────────────────────────────
HF_READY="${HF_HOME}/.${MODEL_NAME//\//_}_downloaded"

if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_NAME}')
print('[cache_lt] snapshot_download done.')
"
    touch "${HF_READY}"
else
    elapsed=0
    until [ -f "${HF_READY}" ]; do
        sleep 10
        elapsed=$((elapsed + 10))
        if [ "${elapsed}" -ge 1800 ]; then
            echo "ERROR: timed out waiting for model download" >&2
            exit 1
        fi
    done
fi

# ── Step B: 并行 cache ────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python cache_activations_last_token_ft.py \
    --model-name           "${MODEL_NAME}" \
    --rank                 "${SLURM_ARRAY_TASK_ID}" \
    --world-size           "${WORLD_SIZE}" \
    --batch-size           "${BATCH_SIZE}" \
    --latentqa-batch-size  "${LATENTQA_BATCH_SIZE}"

echo "[cache_lt] ✓ Task ${SLURM_ARRAY_TASK_ID} 完成"
