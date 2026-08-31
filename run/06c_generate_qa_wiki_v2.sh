#!/bin/bash
#SBATCH --job-name=gen_qa_wiki_v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --array=0-55%8
#SBATCH --output=./logs/qa_gen/%x_%A_%a.out
#SBATCH --error=./logs/qa_gen/%x_%A_%a.err
#SBATCH --exclude=n0046

# ── Regenerate Wikipedia QA with anti-leak prompt (FACTUAL_PROMPT v2) ──────
# Manifest: data/raw/finetune_qa_raw_v2/shards_wiki.jsonl   (shard_size=1000)
#   56 shards = 7 subtypes × 8 shards   (7250 records each, last shard 250)
# Each shard ~7-10 min on 40G A100 MIG (Qwen3-14B bf16) at 1000 records.
# With %8 parallelism: 7 batches × ~8 min ≈ 60 min wall clock.
#
# Submit:
#   sbatch run/06c_generate_qa_wiki_v2.sh
#
# After completion, parse:
#   for s in wikipedia_concept wikipedia_event wikipedia_generic \
#            wikipedia_organization wikipedia_person wikipedia_place wikipedia_work; do
#     python -m scripts.parse_qa \
#         --in-dir data/raw/finetune_qa_raw_v2 \
#         --out-dir data/raw/finetune_qa_diversified_v2 \
#         --subtype "$s"
#   done
#
# Then symlink/copy non-wiki QA from finetune_qa_diversified/ into _v2/
# (skip md_gender.jsonl + ner.jsonl — those are dropped from training mix).
# ───────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

mkdir -p logs/qa_gen

echo "[gen_qa_wiki_v2] task=${SLURM_ARRAY_TASK_ID}  node=${SLURMD_NODENAME}"
nvidia-smi -L

python -m scripts.generate_qa \
    --shard_id "${SLURM_ARRAY_TASK_ID}" \
    --raw-dir  "${PROJ}/data/raw/finetune_diversified_wiki_only" \
    --out-dir  "${PROJ}/data/raw/finetune_qa_raw_v2" \
    --manifest "${PROJ}/data/raw/finetune_qa_raw_v2/shards_wiki.jsonl"

echo "[gen_qa_wiki_v2] task=${SLURM_ARRAY_TASK_ID} done"
