#!/bin/bash
#SBATCH --job-name=cache_reps_extras
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --array=0-3
#SBATCH --output=./logs/data_prep/%x_%A_%a.out
#SBATCH --error=./logs/data_prep/%x_%A_%a.err
#SBATCH --exclude=n0046

# ── Stage 1 _extra sources → Qwen3-4B layer 27 last-token reps ────────────
# 4-way DDP shard (rank 0..3, world_size=4) over ALL *_extra.jsonl files
# Output: data/representations/finetune_diversified_qwen3_l27/
#         {ag_news_extra,tweeteval_sentiment_extra,lmsys_user_extra,
#          dair_emotion_extra}_rank{r}_of_4.npy
#
# 175k total rows across 4 sources.
# Throughput @ 40G MIG: ~55 rows/s/rank → 175k/4 ranks ≈ 13 min compute + 30s load.
# Wall clock: ~15 min (depends on queue).
# ───────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

mkdir -p logs/data_prep

WORLD_SIZE=4

echo "[cache_reps_extras] rank ${SLURM_ARRAY_TASK_ID}/${WORLD_SIZE}  node=${SLURMD_NODENAME}"

# Build a per-task staging dir (avoid races between array tasks setting up
# the same path). Contains symlinks to ONLY the *_extra.jsonl files so
# --all doesn't accidentally re-cache the 17 already-cached sources.
STAGE="${PROJ}/tmp/extras_staging_rank${SLURM_ARRAY_TASK_ID}"
rm -rf "${STAGE}" && mkdir -p "${STAGE}"
for f in "${PROJ}"/data/raw/finetune_diversified/*_extra.jsonl; do
    ln -sf "$f" "${STAGE}/$(basename "$f")"
done
ls -la "${STAGE}"

python experiments/data_prep/cache_finetune_reps.py \
    --rank       "${SLURM_ARRAY_TASK_ID}" \
    --world-size "${WORLD_SIZE}"           \
    --all                                   \
    --raw-dir    "${STAGE}"                 \
    --out-dir    "${PROJ}/data/representations/finetune_diversified_qwen3_l27" \
    --model-name "Qwen/Qwen3-4B"           \
    --layer      27                         \
    --batch-size 2000

echo "[cache_reps_extras] rank ${SLURM_ARRAY_TASK_ID} done"
