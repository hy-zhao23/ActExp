#!/bin/bash
#SBATCH --job-name=selfie_nopatch
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

echo "[selfie_nopatch] node=${SLURMD_NODENAME}"

# Same prompt as smoke, but --no-patch ⇒ hook never fires.
# Compare against selfie_smoke results to see if patch had any effect.
python experiments/eval/baseline_selfie.py \
    --split val \
    --n-samples 5 \
    --inject-layer 3 --num-placeholders 5 \
    --no-patch --variant-name nopatch \
    --sources wikipedia_person \
    --output-dir out/eval/selfie_nopatch

echo "--- nopatch generations (compare with selfie_smoke) ---"
cat out/eval/selfie_nopatch/wikipedia_person.jsonl | python -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    print(f\"Q: {r['question'][:100]}\")
    print(f\"GT: {r['gt'][:100]}\")
    print(f\"NoPatch: {r['nopatch'][:200]}\")
    print()
"
