#!/bin/bash
#SBATCH --job-name=gen_qa_new
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --array=0-15%16
#SBATCH --output=./logs/qa_gen/%x_%A_%a.out
#SBATCH --error=./logs/qa_gen/%x_%A_%a.err
#SBATCH --exclude=n0046

# ── Stage 2 QA generation for finetune_new/ (lmsys_user + dair_emotion) ────
# Manifest: data/raw/finetune_qa_raw/shards_new.jsonl  (shard_size=2500)
#   shards 0-7  : dair_emotion (8 shards × ~2.5k rows = 19,912 total)
#   shards 8-15 : lmsys_user   (8 shards × 2.5k rows = 20,000 total)
# Each shard ~10 min compute + 3 min model load on a 40G A100 MIG (Qwen3-14B bf16).
# With 16-way parallelism: ~13 min wall clock.
#
# Submit:
#   bash run/06b_generate_qa_new.sh        # interactive run (one shard at a time)
#   sbatch run/06b_generate_qa_new.sh      # SLURM array, 2 shards parallel
#
# After completion, run parse:
#   python -m scripts.parse_qa \
#       --in-dir data/raw/finetune_qa_raw \
#       --out-dir data/raw/finetune_qa_new \
#       --subtype dair_emotion
#   python -m scripts.parse_qa \
#       --in-dir data/raw/finetune_qa_raw \
#       --out-dir data/raw/finetune_qa_new \
#       --subtype lmsys_user
# ───────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

mkdir -p logs/qa_gen

echo "[gen_qa_new] task=${SLURM_ARRAY_TASK_ID}  node=${SLURMD_NODENAME}"
nvidia-smi -L

python -m scripts.generate_qa \
    --shard_id "${SLURM_ARRAY_TASK_ID}" \
    --raw-dir  "${PROJ}/data/raw/finetune_new" \
    --out-dir  "${PROJ}/data/raw/finetune_qa_raw" \
    --manifest "${PROJ}/data/raw/finetune_qa_raw/shards_new.jsonl"

echo "[gen_qa_new] task=${SLURM_ARRAY_TASK_ID} done"
