#!/bin/bash
#SBATCH --job-name=stage1_extras
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --qos=standard
#SBATCH --requeue
#SBATCH --array=0-3
#SBATCH --output=./logs/data_prep/%x_%A_%a.out
#SBATCH --error=./logs/data_prep/%x_%A_%a.err
#SBATCH --exclude=n0046

# ── Stage 1 supplements: 4 source extras, parallel array (CPU only) ───────
# task 0 → ag_news_extra              (~30s, complement of seed=42 selection)
# task 1 → tweeteval_sentiment_extra  (~10s, full complement, ~25k rows)
# task 2 → lmsys_user_extra           (~10 min, stream LMSYS skip-on-hash)
# task 3 → dair_emotion_extra         (~2 min, load dair unsplit 416k filter)
#
# All 4 tasks run independently; wall clock ≈ max(any task) ≈ 10 min.
# No GPU needed; HF datasets cached at $HF_HOME.
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

mkdir -p logs/data_prep

echo "[stage1_extras] task=${SLURM_ARRAY_TASK_ID}  node=${SLURMD_NODENAME}"

python experiments/data_prep/sample_stage1_extras.py \
    --task-id "${SLURM_ARRAY_TASK_ID}"

echo "[stage1_extras] task=${SLURM_ARRAY_TASK_ID} done"
