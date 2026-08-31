#!/bin/bash
#SBATCH --job-name=ao_eval_lqa_nolora
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --qos=standard
#SBATCH --output=./logs/gpu/%x_%j.out
#SBATCH --error=./logs/gpu/%x_%j.err
#SBATCH --exclude=n0046

# Ablation: same eval but verbalizer_lora_path=None (base model only)
# 对比 finetuned verbalizer 和 base model 在 latentqa holdout 上的差异

set -euo pipefail

export PROJ="${PROJ:-$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
source "${PROJ}/setup_env.sh"

CODE_DIR="${PROJ}/experiments/baseline/activation_oracles"
cd "$CODE_DIR"

python3 - <<'EOF'
import os, json, random, torch
from peft import LoraConfig
import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerInputInfo
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.dataset_classes.misc import latentqa_loader

SEED = 42
random.seed(SEED); torch.manual_seed(SEED)

paths = latentqa_loader.DataPaths(
    system=None,
    stimulus_completion='datasets/latentqa_datasets/eval/stimulus_completion.json',
    stimulus='datasets/latentqa_datasets/eval/stimulus.json',
    control='datasets/latentqa_datasets/eval/control.json',
    qa='datasets/latentqa_datasets/eval/qa.json',
)
dataset = latentqa_loader.load_latentqa_dataset(paths, filter_prefixes=[], train_percent=1.0, seed=SEED)

by_source = {'control': [], 'stimulus': [], 'stimulus_completion': []}
for i in range(len(dataset)):
    item = dataset[i]
    if item['source'] in by_source:
        by_source[item['source']].append(item)

rng = random.Random(SEED)
by_source = {s: rng.sample(v, min(20, len(v))) for s, v in by_source.items()}

device = torch.device('cuda')
dtype = torch.bfloat16
torch.set_grad_enabled(False)

tokenizer = load_tokenizer('Qwen/Qwen3-4B')
model = load_model('Qwen/Qwen3-4B', dtype)
model.eval()
model.add_adapter(LoraConfig(), adapter_name='default')

generation_kwargs = {'do_sample': False, 'temperature': None, 'top_p': None, 'max_new_tokens': 40}
add_gen_prompt = {'stimulus_completion': False, 'stimulus': True, 'control': True}

results_dir = 'experiments/latentqa_results'
lines = []
for source, items in by_source.items():
    print(f'[eval] source={source}  n={len(items)}', flush=True)
    config = base_experiment.VerbalizerEvalConfig(
        model_name='Qwen/Qwen3-4B',
        activation_input_types=['orig'],
        eval_batch_size=64,
        verbalizer_generation_kwargs=generation_kwargs,
        verbalizer_input_types=['full_seq'],
        full_seq_repeats=1,
        segment_repeats=1,
        add_generation_prompt=add_gen_prompt[source],
    )
    prompt_infos = [VerbalizerInputInfo(
        context_prompt=item['read_prompt'],
        verbalizer_prompt=item['dialog'][0]['content'],
        ground_truth=item['dialog'][1]['content'],
    ) for item in items]

    results = base_experiment.run_verbalizer(
        model=model, tokenizer=tokenizer,
        verbalizer_prompt_infos=prompt_infos,
        verbalizer_lora_path=None,
        target_lora_path=None,
        config=config, device=device,
    )

    lines.append(f'\n=== {source} (n={len(results)}, NO verbalizer LoRA) ===\n')
    for i, r in enumerate(results):
        pred = r.full_sequence_responses[0] if r.full_sequence_responses else '(empty)'
        lines.append(f'[{i:2d}] Q   : {r.verbalizer_prompt}')
        lines.append(f'     GT  : {r.ground_truth}')
        lines.append(f'     PRED: {pred}')
        lines.append('')

out = '\n'.join(lines)
with open(f'{results_dir}/summary_no_lora.txt', 'w') as f:
    f.write(out)
print(out)
print('[eval] done -> experiments/latentqa_results/summary_no_lora.txt')
EOF
