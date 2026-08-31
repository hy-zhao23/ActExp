#!/bin/bash
#SBATCH --job-name=gen_qa
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --output=./logs/qa_gen/%x_%A_%a.out
#SBATCH --error=./logs/qa_gen/%x_%A_%a.err
#SBATCH --exclude=n0046

# ── Stage 2 QA generation (shard-based) ────────────────────────────────────
# Each array task runs ONE shard (20K rows) on a 40G A100 MIG slice.
# Model: Qwen3-14B bf16  (27.5 GB weights + 6.5 GB KV fits 42.4 GB slice)
# Throughput: ~4.8 req/s → 20K shard ≈ 70 min gen + 3 min load = ~75 min.
# 42 shards × %16 concurrency ≈ 3 h wall clock.
#
# Preemption-safe: qos=low + --requeue; each shard writes raw JSONL appended
# every 100 records — restart resumes from last flushed text_idx.
#
# BEFORE SUBMITTING: run `python -m scripts.build_shards` once to build the
# manifest, read its "total shards" line, and set --array=0-(N-1)%16 below.
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

mkdir -p logs/qa_gen

echo "[gen_qa] task=${SLURM_ARRAY_TASK_ID}  node=${SLURMD_NODENAME}"
nvidia-smi -L
python -m scripts.generate_qa --shard_id "${SLURM_ARRAY_TASK_ID}"
echo "[gen_qa] task=${SLURM_ARRAY_TASK_ID} done"
