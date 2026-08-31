# Universal Activation Verbalizer (UAV)

UAV decodes a language model's last-token activation into text. Training has
two stages: activation-to-text alignment, followed by QA fine-tuning with a
frozen decoder and LoRA.

**Paper:** [Universal Activation Verbalizer: A Unified Framework for
Cross-Model Activation Explanation](https://arxiv.org/abs/2605.25903)
([arXiv:2605.25903](https://arxiv.org/abs/2605.25903))

**Checkpoints:** [Hugging Face](https://huggingface.co/adfjasigageadfd/uav)

## Setup

The recorded environment uses Python 3.10, PyTorch 2.7.1, and CUDA 12.6.

```bash
conda create -n oracle python=3.10
conda activate oracle
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

The provided launchers target a SLURM cluster and derive `PROJ` from the Git
repository root. Activate your environment and load cluster modules before
submitting jobs; export `PROJ` first only if you need to override that root.

## Data

Prepare the following directories:

```text
data/raw/finetune_diversified/
data/raw/finetune_qa_diversified/
data/raw/finetune_qa_diversified_val/
data/raw/finetune_qa_diversified_test/
data/representations/finetune_diversified/
```

Data construction and representation-caching utilities are under
`experiments/data_prep/` and `scripts/`. Run `python -m scripts.build_splits`
after generating the diversified QA data.

## Train and Evaluate

```bash
# Stage 1: activation-to-text alignment
sbatch run/data_scale/03_pretrain_v1-2_500k_diverse.sh

# Stage 2: QA fine-tuning
sbatch run/data_scale/13_finetune_v1-2_500k_diverse_fullqa.sh

# Generate and score the main-model evaluation outputs
sbatch --array=3 scripts/eval/eval_gen_ours_array.sh manifest_ours_40g.tsv
sbatch scripts/eval/eval_score_all.sh
```

Checkpoints are written to `checkpoints/`, evaluation outputs to `out/eval/`,
and logs to `logs/`. The main training configuration is under
`experiments/training/configs/data_scale/`.
