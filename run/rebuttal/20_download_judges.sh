#!/bin/bash
#SBATCH --job-name=reb_dl_judges
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --qos=low
#SBATCH --time=8:00:00
#SBATCH --output=./logs/cpu/%x_%j.out
#SBATCH --error=./logs/cpu/%x_%j.err

# R4: 单进程预下载三个 judge 模型（教训 R1：并发下载会写坏 HF cache）

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

for m in "Qwen/Qwen2.5-14B-Instruct" "google/gemma-3-27b-it" "meta-llama/Llama-3.3-70B-Instruct"; do
    echo "[download] $m"
    hf download "$m" --exclude "*.pth" --exclude "original/*" --exclude "*.gguf"
done
echo "[download] all judges done"
