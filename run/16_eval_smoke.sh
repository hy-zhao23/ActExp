#!/bin/bash
#SBATCH --job-name=eval_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --qos=standard
#SBATCH --time=1:30:00
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# Smoke test: run ours (v1-2_2M_lr1e-4 step_5200) + AO (lr1e-4 best) on the
# 9 open-generation sources × 50 samples, then score with EM/RougeL/BERTScore.
# 20G MIG GPU is enough: Qwen3-4B bf16 ≈ 8G + LoRA + KV cache + bertscore model.

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

OUT_OURS="${PROJ}/out/eval/ours_v1-2_2M_lr1e-4_step5200"
OUT_AO="${PROJ}/out/eval/ao_qa_lr1e-4"

# ── ours ──────────────────────────────────────────────────────────────────────
echo "[smoke] running OURS gen → ${OUT_OURS}"
python experiments/eval/eval_gen_ours.py \
    --config       experiments/training/configs/qwen3_4b_v1-2_2M_lr1e-4.yaml \
    --stage1-ckpt  checkpoints/oracle_act_qwen3_4b_v1-2/ep16/final \
    --exact-ckpts  ours_step5200:checkpoints/oracle_ft_qwen3_4b_v1-2_2M_lr1e-4/step_5200 \
    --n-samples    50 \
    --output-dir   "${OUT_OURS}"

# ── AO ────────────────────────────────────────────────────────────────────────
echo "[smoke] running AO gen → ${OUT_AO}"
python experiments/eval/eval_gen_ao.py \
    --ao-dirs    ao_lr1e-4:${PROJ}/checkpoints/ao_stage2_qa_ft_Qwen3-4B_lr1e-4 \
    --n-samples  50 \
    --output-dir "${OUT_AO}"

# ── score both ────────────────────────────────────────────────────────────────
echo "[smoke] scoring both → metrics_summary.md in each dir"
python experiments/eval/eval_score.py \
    --eval-dir "${OUT_OURS}" \
    --eval-dir "${OUT_AO}" \
    --bertscore-model microsoft/deberta-xlarge-mnli

echo "[smoke] ✓ done"
