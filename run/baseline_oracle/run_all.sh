#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Activation Oracles — 论文实验完整复现流水线
#
# 用法（在项目根目录执行）：
#   Run from the repository root.
#   bash run/run_all.sh [--skip-cache] [--skip-train]
#
# 默认模式：预缓存激活 → 训练 → 评测 → 绘图（依次提交，后续 job 依赖前一步）
# --skip-cache：跳过预缓存（直接用 on-the-fly 激活收集，较慢）
# --skip-train：跳过训练，直接用 HuggingFace 上的预训练权重评测
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SKIP_CACHE=false
SKIP_TRAIN=false
for arg in "$@"; do
    [[ "$arg" == "--skip-cache" ]] && SKIP_CACHE=true
    [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=true
done

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
cd "$PROJ"

echo "========================================================"
echo "  Activation Oracles — 实验复现流水线"
echo "  预缓存: $([ "$SKIP_CACHE" = true ] && echo '跳过' || echo '启用（4×A100 并行）')"
echo "  训练:   $([ "$SKIP_TRAIN" = true ] && echo '跳过（使用预训练权重）' || echo '完整流程')"
echo "========================================================"

# ── Step 0: 预缓存激活向量（4 GPU job array，约 4~5 小时）────
CACHE_DEP=""
if [ "$SKIP_CACHE" = false ] && [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "[0/3] 提交预缓存 job array..."
    CACHE_JOB=$(sbatch --parsable run/00_cache.sh)
    echo "      job array ID: $CACHE_JOB  (4×A100 并行，partition=gpu)"
    CACHE_DEP="--dependency=afterok:${CACHE_JOB}"
else
    echo ""
    echo "[0/3] 跳过预缓存"
fi

# ── Step 1: 训练 ────────────────────────────────────────────
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "[1/3] 提交训练 job..."
    TRAIN_JOB=$(sbatch --parsable ${CACHE_DEP} run/01_train.sh)
    echo "      job ID: $TRAIN_JOB  (partition=gpu, 4×A100, ~24h)"
    TRAIN_DEP="--dependency=afterok:${TRAIN_JOB}"
else
    echo ""
    echo "[1/3] 跳过训练，使用 HuggingFace 预训练权重（adamkarvonen/checkpoints_*）"
    TRAIN_DEP=""
fi

# ── Step 2: 评测（三个 job 并行，均依赖训练完成） ──────────
echo ""
echo "[2/3] 提交评测 jobs（${SKIP_TRAIN:+无依赖，}并行运行）..."

CLS_JOB=$(sbatch --parsable ${TRAIN_DEP} run/02_eval_classification.sh)
echo "      Classification job ID: $CLS_JOB"

SK_JOB=$(sbatch --parsable ${TRAIN_DEP} run/02_eval_secret_keeping.sh)
echo "      Secret Keeping  job ID: $SK_JOB  (Gender + Taboo + SSC)"

PQA_JOB=$(sbatch --parsable ${TRAIN_DEP} run/02_eval_personaqa.sh)
echo "      PersonaQA       job ID: $PQA_JOB"

# ── Step 3: 绘图（依赖全部评测完成） ──────────────────────
echo ""
echo "[3/3] 提交绘图 job（等待全部评测完成后启动）..."
PLOT_DEP="--dependency=afterok:${CLS_JOB}:${SK_JOB}:${PQA_JOB}"
PLOT_JOB=$(sbatch --parsable ${PLOT_DEP} run/03_plots.sh)
echo "      Plots job ID: $PLOT_JOB"

# ── 汇总 ───────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  已提交 job 汇总："
[ "$SKIP_CACHE" = false ] && [ "$SKIP_TRAIN" = false ] && echo "    预缓存 (×4)    → $CACHE_JOB"
[ "$SKIP_TRAIN" = false ] && echo "    训练           → $TRAIN_JOB"
echo "    分类评测         → $CLS_JOB"
echo "    Secret Keeping   → $SK_JOB"
echo "    PersonaQA        → $PQA_JOB"
echo "    论文图表         → $PLOT_JOB"
echo ""
echo "  监控命令："
echo "    squeue -u \$USER"
[ "$SKIP_TRAIN" = false ] && echo "    tail -f logs/gpu/ao_train_${TRAIN_JOB:-*}.out  # 训练日志"
echo "========================================================"
