#!/bin/bash
# ────────────────────────────────────────────────────────────
# Submit baseline_cls_test jobs in dependency order:
#   Step 1: cache reps (array 0-6, 7 parallel tasks)
#   Step 2: finetune  (starts only after ALL cache tasks succeed)
#
# 用法:
#   bash run/baseline_cls_test/submit.sh
#   bash run/baseline_cls_test/submit.sh --stage1-ckpt <path>
# ────────────────────────────────────────────────────────────

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
STAGE1_CKPT="${PROJ}/checkpoints/oracle_act_qwen3_4b/step_2200"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage1-ckpt) STAGE1_CKPT="$2"; shift 2 ;;
        *) echo "[warn] unknown arg: $1"; shift ;;
    esac
done

cd "$PROJ"

# Step 1: cache reps (array 0-6)
CACHE_JID=$(sbatch --parsable run/baseline_cls_test/01_cache_reps.sh)
echo "[submit] Step 1 cache  → job array ${CACHE_JID}"

# Step 2: finetune — start only after all array tasks of Step 1 succeed
FT_JID=$(sbatch --parsable \
    --dependency=afterok:${CACHE_JID} \
    run/baseline_cls_test/02_finetune.sh \
        --stage1-ckpt "${STAGE1_CKPT}")
echo "[submit] Step 2 finetune → job ${FT_JID}  (depends on ${CACHE_JID})"

echo ""
echo "Monitor:"
echo "  squeue -j ${CACHE_JID},${FT_JID}"
echo "  tail -f logs/gpu/bcls_ft_${FT_JID}.out"
