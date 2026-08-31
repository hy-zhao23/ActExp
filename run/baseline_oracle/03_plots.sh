#!/bin/bash
#SBATCH --job-name=ao_plots
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --qos=standard
#SBATCH --output=./logs/cpu/%x_%j.out
#SBATCH --error=./logs/cpu/%x_%j.err
#SBATCH --exclude=n0046

# ────────────────────────────────────────────────────────────
# Step 3: 生成论文全部图表
# Run from the repository root: sbatch run/baseline_oracle/03_plots.sh
#
# 依赖：Step 2 的评测结果 JSON 已存在
# 输出：experiments/activation_oracles/experiments/final_paper_plots/
# ────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"

CODE_DIR="${PROJ}/experiments/activation_oracles"
cd "$CODE_DIR"
echo "[plots] 工作目录: $CODE_DIR"

PLOTS_DIR="experiments/final_paper_plots"

echo "[plots] 运行所有论文绘图脚本..."
bash "${PLOTS_DIR}/run_all_plots.sh"

echo "[plots] ✓ 图表生成完成，见 ${PLOTS_DIR}/"
