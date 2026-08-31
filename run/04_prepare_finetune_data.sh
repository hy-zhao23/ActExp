#!/bin/bash
#SBATCH --job-name=prep_finetune
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --qos=standard
#SBATCH --requeue                         # auto-restart if preempted (qos=low risk)
#SBATCH --output=./logs/data_prep/%x_%j.out
#SBATCH --error=./logs/data_prep/%x_%j.err
#SBATCH --exclude=n0046

# ── Stage 2 finetune data preparation ──────────────────────────────────────
# Filters all source datasets → data/raw/finetune/{source}_{subtype}.jsonl
#
# Sources (sequential; each independent):
#   wikipedia  ~1.2M  (split into 7 subtypes)
#   scientific ~800k
#   ag_news    ~120k  (HF download on first run)
#   tweeteval  ~70k   (HF download; 3 subsets)
#   sst2/md_gender/ner  (local CSVs in data/raw/finetune_v0_archive/)
#   latentqa   ~16k
#
# Usage:
#   sbatch run/04_prepare_finetune_data.sh              # all sources, full data
#   sbatch run/04_prepare_finetune_data.sh --limit 5000 # smoke test
#   sbatch run/04_prepare_finetune_data.sh wikipedia    # single source
#
# Estimated runtime: ~1.5-2h for full run (dominated by Qwen3 tokenizer on wiki+sci).
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

mkdir -p logs/data_prep

# parse args: optional --limit N and/or single source name
SOURCE="all"
LIMIT_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit) LIMIT_ARG="--limit $2"; shift 2 ;;
        *)       SOURCE="$1";           shift ;;
    esac
done

echo "[prep] node=${SLURMD_NODENAME}  source=${SOURCE}  ${LIMIT_ARG}"
echo "[prep] output: ${PROJ}/data/raw/finetune/"

python -m scripts.prepare_data --source "${SOURCE}" ${LIMIT_ARG}

echo "[prep] done"
ls -la "${PROJ}/data/raw/finetune/" | tail -20
