#!/bin/bash
#SBATCH --job-name=selfie_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=standard
#SBATCH --time=00:20:00
#SBATCH --exclude=n0046
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err

set -euo pipefail
export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[selfie_smoke] node=${SLURMD_NODENAME}"

# 5 samples × 1 layer × 1 source = pipeline sanity check
python experiments/eval/baseline_selfie.py \
    --split val \
    --n-samples 5 \
    --inject-layer 3 \
    --num-placeholders 5 \
    --sources wikipedia_person \
    --output-dir out/eval/selfie_smoke

echo "[selfie_smoke] done"
ls -la out/eval/selfie_smoke/
echo "--- generations ---"
cat out/eval/selfie_smoke/wikipedia_person.jsonl | python -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    print(f\"Q: {r['question'][:100]}\")
    print(f\"GT: {r['gt'][:100]}\")
    print(f\"SelfIE: {r['selfie'][:200]}\")
    print()
"
