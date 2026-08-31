#!/bin/bash
#SBATCH --job-name=reb_eval_gen_r2_qatypes
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --time=8:00:00
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004

# R2 下游 gen-eval：gistcls / factonly / mix500k(seed42) 三个 ft ckpt 同目录三列对比。
# 三者共享 stage-1 (1026779) 与 ft hp，唯一差异 = stage-2 数据类型 → 生成指标可直接归因。
# loss 版 3×2 矩阵见 exp_rebuttal.md R2；本 job 补 R-L/BERTS 口径（reviewer 更认生成指标）。

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

CONFIG="${PROJ}/experiments/training/configs/rebuttal/qwen3_4b_instr_self_500k.yaml"
STAGE1_CKPT="${PROJ}/checkpoints/oracle_act_qwen3_4b_l27_diversified/final"
CK="${PROJ}/checkpoints/rebuttal"
OUTPUT_DIR="${PROJ}/out/eval/rebuttal_r2_qatypes"

for p in "$CONFIG" "$STAGE1_CKPT" "$CK/oracle_ft_qwen3_instr_self_500k_gistcls" \
         "$CK/oracle_ft_qwen3_instr_self_500k_factonly" "$CK/oracle_ft_qwen3_instr_self_500k"; do
    [[ -e "$p" ]] || { echo "[error] missing: $p" >&2; exit 1; }
done

N_SAMPLES="${N_SAMPLES:-100}"

echo "[r2_qatypes] node=${SLURMD_NODENAME}  output=${OUTPUT_DIR}"

python experiments/eval/eval_gen_ours.py \
    --config       "${CONFIG}" \
    --stage1-ckpt  "${STAGE1_CKPT}" \
    --ft-dirs      "r2_gistcls:${CK}/oracle_ft_qwen3_instr_self_500k_gistcls" \
                   "r2_factonly:${CK}/oracle_ft_qwen3_instr_self_500k_factonly" \
                   "r2_mix500k:${CK}/oracle_ft_qwen3_instr_self_500k" \
    --n-samples    "${N_SAMPLES}" \
    --output-dir   "${OUTPUT_DIR}"

echo "[r2_qatypes] ✓ done → ${OUTPUT_DIR}"
