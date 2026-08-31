#!/bin/bash
#SBATCH --job-name=cache_qwen3_layer_abl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --array=0-3
#SBATCH --output=./logs/data_prep/%x_%A_%a.out
#SBATCH --error=./logs/data_prep/%x_%A_%a.err
#SBATCH --exclude=n0046,n0004

# ── Qwen3-4B layer ablation cache (diversified finetune data) ─────────────
# Mirrors L27 build (05c+05d combined into one job over the full
# data/raw/finetune_diversified dir).
#
# Submit with:  sbatch --export=ALL,LAYER=<idx> run/17_cache_diverse_qwen3_layer_ablation.sh
# LAYER = 0-indexed model.model.layers[i] (i.e. the same convention as L27).
#
# Output: data/representations/finetune_diversified_qwen3_l${LAYER}/
#         {source}_rank{r}_of_4.npy   + metadata.json
#
# Total rows ≈ 470k → ~36 min / rank @ 40G MIG → 2h limit is comfortable.
# ───────────────────────────────────────────────────────────────────────────

set -euo pipefail

if [[ -z "${LAYER:-}" ]]; then
    echo "[error] LAYER env var not set. Submit with --export=ALL,LAYER=<idx>" >&2
    exit 1
fi

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

mkdir -p logs/data_prep

WORLD_SIZE=4
OUT_DIR="${PROJ}/data/representations/finetune_diversified_qwen3_l${LAYER}"

echo "[cache_layer_abl] LAYER=${LAYER}  rank ${SLURM_ARRAY_TASK_ID}/${WORLD_SIZE}  node=${SLURMD_NODENAME}"
echo "[cache_layer_abl] out_dir=${OUT_DIR}"

python experiments/data_prep/cache_finetune_reps.py \
    --rank       "${SLURM_ARRAY_TASK_ID}" \
    --world-size "${WORLD_SIZE}"           \
    --all                                   \
    --raw-dir    "${PROJ}/data/raw/finetune_diversified" \
    --out-dir    "${OUT_DIR}"               \
    --model-name "Qwen/Qwen3-4B"           \
    --layer      "${LAYER}"                 \
    --batch-size 2000

echo "[cache_layer_abl] LAYER=${LAYER} rank ${SLURM_ARRAY_TASK_ID} done"
