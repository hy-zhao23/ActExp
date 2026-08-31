#!/bin/bash
#SBATCH --job-name=ao_cache
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1    # 40G MIG 切片；hook 已 CPU-offload，峰值 ~26G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --qos=standard
#SBATCH --output=./logs/gpu/%x_%A_%a.out
#SBATCH --error=./logs/gpu/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --exclude=n0046

# ────────────────────────────────────────────────────────────
# Step 0: 预缓存激活向量（4×A100 40G MIG 并行，可在训练前提交）
#
# 用途：
#   LatentQA、PastLens、Classification(train) 默认存 context_input_ids，
#   训练时实时推理取激活（on-the-fly）。该脚本提前算好写回磁盘，训练
#   时直接读取，无需额外前向传播，显著提升 GPU 利用率。
#
# 提交：
#   Run from the repository root: sbatch run/baseline_oracle/00_cache.sh
#
# 断点续传：
#   每个 loader 完成后写 .cache_done 标记；重新提交时 O(1) 跳过已完成的，
#   被 kill 中途的那一个从头重跑（一个 loader ≤ 10 分钟）。
#
# 资源说明（Qwen3-4B bfloat16，40G MIG，hook 已立即 offload 到 CPU）：
#   模型 8 GB，峰值公式：8 + B×L×38,912 bytes
#   其中 38912 = (4×D + 2×I)×2/2 = 实测 FFN gate+up 层峰值（bfloat16）
#   峰值时刻：层 27 FFN 期间 gate+up 同时在显存（hook tensor 已在 CPU 不占 GPU）
#   PastLens/Classification（L≤1800）：B=256 → ~26 GB（67%）✓ 40G
#   LatentQA（L 可达 4000+）：          B=64  → ~18 GB（46%）✓ 40G
#   注：原 80G 方案 B=2048/128，OOM 原因是 hook tensor 未及时 offload，现已修复
#
# 修改模型：
#   在下方 MODEL_NAME 处改，需与 sft.py 的 models 列表一致。
# ────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"

CODE_DIR="${PROJ}/experiments/baseline/activation_oracles"
cd "$CODE_DIR"

MODEL_NAME="Qwen/Qwen3-4B"   # ← 与 sft.py 的 models 列表保持一致
WORLD_SIZE=4                  # ← 与 --array=0-<N-1> 的 N 保持一致
BATCH_SIZE=256                # PastLens / Classification（L≤1800）→ 峰值 ~26 GB（67%）✓ 40G
LATENTQA_BATCH_SIZE=64        # LatentQA（L 可达 4000+）         → 峰值 ~18 GB（46%）✓ 40G

echo "[cache] Task ${SLURM_ARRAY_TASK_ID}/${WORLD_SIZE}  GPU: ${CUDA_VISIBLE_DEVICES}"
echo "[cache] 模型: ${MODEL_NAME}  short_bs=${BATCH_SIZE}  latentqa_bs=${LATENTQA_BATCH_SIZE}"

# ── Step A: 序列化 HF 模型下载，避免 4 个 task 并发写坏 JSON ───────────────
# 4 个 array task 同时启动；task 0 负责下载，其余 task 等待标记文件出现。
# 若上次已下载（标记存在），所有 task 直接跳过此步。
HF_READY="${HF_HOME}/.${MODEL_NAME//\//_}_downloaded"

if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
    echo "[cache] Task 0: caching model weights to ${HF_HOME} ..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_NAME}')
print('[cache] snapshot_download done.')
"
    touch "${HF_READY}"
    echo "[cache] Model ready, marker written: ${HF_READY}"
else
    echo "[cache] Task ${SLURM_ARRAY_TASK_ID}: waiting for task 0 model download..."
    elapsed=0
    until [ -f "${HF_READY}" ]; do
        sleep 10
        elapsed=$((elapsed + 10))
        if [ "${elapsed}" -ge 1800 ]; then
            echo "ERROR: timed out waiting for model download (30 min)" >&2
            exit 1
        fi
    done
    echo "[cache] Task ${SLURM_ARRAY_TASK_ID}: model ready."
fi

# ── Step B: 并行 cache 激活向量 ───────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python -m nl_probes.cache_activations \
    --model-name           "${MODEL_NAME}" \
    --rank                 "${SLURM_ARRAY_TASK_ID}" \
    --world-size           "${WORLD_SIZE}" \
    --batch-size           "${BATCH_SIZE}" \
    --latentqa-batch-size  "${LATENTQA_BATCH_SIZE}"

echo "[cache] ✓ Task ${SLURM_ARRAY_TASK_ID} 完成"
