#!/bin/bash
#SBATCH --job-name=reb_llm_judge
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046,n0004

# R4: LLM-judge (TOPIC/DETAILS 0-5) on ours_self_qwen3_4b, 100 slots.
# 单 judge 单作业（好 backfill）：JUDGE 由 --export 传入；TP=可见 GPU 数。
#   sbatch -J reb_judge_qwen14b  --export=ALL,JUDGE=qwen14b  run/rebuttal/21_llm_judge.sh
#   sbatch -J reb_judge_gemma27b --export=ALL,JUDGE=gemma27b run/rebuttal/21_llm_judge.sh
#   sbatch -J reb_judge_llama70b --gres=gpu:a100:4 --mem=128G --cpus-per-task=16 \
#          --export=ALL,JUDGE=llama70b run/rebuttal/21_llm_judge.sh
# 三个都完成后（登录节点即可）：python rebuttal/llm_judge.py --aggregate

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"
cd "$PROJ"

# oracle env 的 transformers 5.x 与 vllm 0.10 不兼容（1097806-15 三连挂）；
# verl env 是自洽的 vllm 0.11 + transformers 4.56，judge 用它。
micromamba activate verl

export VLLM_WORKER_MULTIPROC_METHOD=spawn

SET="${SET:-orig}"
VS="${VS:-passage}"
echo "[judge] ${JUDGE} set=${SET} vs=${VS} start $(date +%H:%M) gpus=${CUDA_VISIBLE_DEVICES:-?}"
python rebuttal/llm_judge.py --judge "${JUDGE}" --set "${SET}" --vs "${VS}"
echo "[reb_llm_judge:${JUDGE}:${SET}:${VS}] done"
