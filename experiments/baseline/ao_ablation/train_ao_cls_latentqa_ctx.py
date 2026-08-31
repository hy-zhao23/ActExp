"""
AO training: Classification + LatentQA + PastLens (ctx), last-token, layer 75%.
Follows original nl_probes/sft.py settings (batch_size=16, lora r=64/alpha=128, lr=1e-5).

"ctx" = PastLens dataset: given activation at position T, predict surrounding tokens.
Uses max_k_activations=1 (single-token injection) to stay consistent with last-token setting.

Launch (single GPU):
    torchrun --nproc_per_node=1 tests/ao/train_ao_cls_latentqa_ctx.py
"""

import os
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.chdir(Path(__file__).resolve().parents[1] / "activation_oracles")

import gc
import math
import random
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader, DatasetLoaderConfig
from nl_probes.dataset_classes.classification import ClassificationDatasetConfig, ClassificationDatasetLoader
from nl_probes.dataset_classes.latentqa_dataset import LatentQADatasetConfig, LatentQADatasetLoader
from nl_probes.dataset_classes.past_lens_dataset import PastLensDatasetConfig, PastLensDatasetLoader
from nl_probes.utils.activation_utils import get_hf_submodule, get_text_only_lora_targets
from nl_probes.utils.common import load_model, load_tokenizer, set_seed
from nl_probes.utils.dataset_utils import BatchData, TrainingDataPoint, construct_batch, materialize_missing_steering_vectors
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook

EXP = "[ao_cls_latentqa_ctx]"


# ── helpers ───────────────────────────────────────────────────────────────────

def train_features_batch(cfg, batch, model, submodule, device, dtype):
    hook_fn = get_hf_activation_steering_hook(
        vectors=batch.steering_vectors,
        positions=batch.positions,
        steering_coefficient=cfg.steering_coefficient,
        device=device,
        dtype=dtype,
    )
    with add_hook(submodule, hook_fn):
        loss = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels).loss
    return loss


def eval_all_datasets(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step):
    model.eval()
    with torch.no_grad():
        cls_data = [dp for ds, data in eval_datasets.items() if ds not in ("latentqa", "past_lens") for dp in data]
        if cls_data:
            total, n = 0.0, 0
            for start in range(0, len(cls_data), cfg.eval_batch_size):
                batch_list = cls_data[start : start + cfg.eval_batch_size]
                batch_list = materialize_missing_steering_vectors(batch_list, tokenizer, model)
                batch = construct_batch(batch_list, tokenizer, device)
                total += train_features_batch(cfg, batch, model, submodule, device, dtype).item()
                n += 1
            avg = total / n
            print(f"{EXP} [val step {global_step}] cls (all) | loss={avg:.4f}  ppl={math.exp(min(avg, 20)):.2f}", flush=True)
        for ds_key in ("latentqa", "past_lens"):
            if ds_key in eval_datasets:
                total, n = 0.0, 0
                for start in range(0, len(eval_datasets[ds_key]), cfg.train_batch_size):
                    batch_list = eval_datasets[ds_key][start : start + cfg.train_batch_size]
                    batch_list = materialize_missing_steering_vectors(batch_list, tokenizer, model)
                    batch = construct_batch(batch_list, tokenizer, device)
                    total += train_features_batch(cfg, batch, model, submodule, device, dtype).item()
                    n += 1
                avg = total / n
                print(f"{EXP} [val step {global_step}] {ds_key:10s} | loss={avg:.4f}  ppl={math.exp(min(avg, 20)):.2f}", flush=True)
    model.train()
    torch.cuda.empty_cache()
    gc.collect()


def length_grouped_reorder(data, batch_size, window_mult=20):
    lengths = [len(d.input_ids) for d in data]
    indices = list(range(len(data)))
    mega = window_mult * batch_size
    megabatches = [indices[i : i + mega] for i in range(0, len(indices), mega)]
    megabatches = [sorted(mb, key=lambda i: lengths[i], reverse=True) for mb in megabatches]
    return [data[i] for i in (j for mb in megabatches for j in mb)]


def build_datasets(cfg, dataset_loaders, latentqa_val_size=500, past_lens_val_size=500, max_len_percentile=0.999):
    set_seed(cfg.seed)
    all_training_data: list[TrainingDataPoint] = []
    all_eval_data: dict[str, list[TrainingDataPoint]] = {}
    latentqa_train: list[TrainingDataPoint] = []
    past_lens_train: list[TrainingDataPoint] = []

    for loader in dataset_loaders:
        split_data = loader.load_dataset("train") if "train" in loader.dataset_config.splits else []
        ds_name = loader.dataset_config.dataset_name
        if ds_name == "latentqa":
            latentqa_train = split_data
        elif ds_name == "past_lens":
            past_lens_train = split_data
        else:
            all_training_data.extend(split_data)
        if "test" in loader.dataset_config.splits:
            all_eval_data[ds_name] = loader.load_dataset("test")

    if latentqa_train:
        all_eval_data["latentqa"] = latentqa_train[:latentqa_val_size]
        all_training_data.extend(latentqa_train[latentqa_val_size:])
    if past_lens_train:
        all_eval_data["past_lens"] = past_lens_train[:past_lens_val_size]
        all_training_data.extend(past_lens_train[past_lens_val_size:])

    lengths = sorted(len(td.input_ids) for td in all_training_data)
    idx = int((len(lengths) - 1) * max_len_percentile)
    threshold = lengths[idx]
    before = len(all_training_data)
    all_training_data = [td for td in all_training_data if len(td.input_ids) <= threshold]
    print(f"Percentile trim: kept <= {threshold} tokens. Removed {before - len(all_training_data)}/{before}.")
    set_seed(cfg.seed)
    random.shuffle(all_training_data)
    all_training_data = length_grouped_reorder(all_training_data, cfg.train_batch_size)
    return all_training_data, all_eval_data


def mk_cfg(custom_params, *, num_train, num_test, splits, model_name, layer_percents, save_acts, batch_size):
    return DatasetLoaderConfig(
        custom_dataset_params=custom_params,
        num_train=num_train,
        num_test=num_test,
        splits=splits,
        model_name=model_name,
        layer_percents=layer_percents,
        save_acts=save_acts,
        batch_size=batch_size,
    )


def train_model(cfg, training_data, eval_datasets, tokenizer, device, dtype, model_kwargs):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    model_kwargs = {**model_kwargs, "device_map": {"": f"cuda:{local_rank}"}}
    set_seed(cfg.seed)
    model = load_model(cfg.model_name, dtype, **model_kwargs)
    model.enable_input_require_grads()
    if cfg.gradient_checkpointing:
        model.use_cache = False
        model.gradient_checkpointing_enable()

    submodule = get_hf_submodule(model, cfg.hook_onto_layer)

    if cfg.load_lora_path is None:
        target_modules = cfg.lora_target_modules
        vlm_targets = get_text_only_lora_targets(cfg.model_name)
        if vlm_targets and target_modules == "all-linear":
            target_modules = vlm_targets
        lora_config = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config, autocast_adapter_dtype=True)
    else:
        model = PeftModel.from_pretrained(model, Path(cfg.load_lora_path), is_trainable=True, autocast_adapter_dtype=True)

    model.print_trainable_parameters()

    torch.cuda.set_device(local_rank)
    train_module = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False,
    )
    train_module.train()

    set_seed(cfg.seed)
    optimizer = torch.optim.AdamW(train_module.parameters(), lr=cfg.lr)

    global_step_size = cfg.train_batch_size * world_size
    effective_steps = (len(training_data) // global_step_size) * global_step_size
    training_data = training_data[:effective_steps][rank::world_size]
    num_batches = len(training_data) // cfg.train_batch_size
    batches = (num_batches // cfg.gradient_accumulation_steps) * cfg.gradient_accumulation_steps
    training_data = training_data[: batches * cfg.train_batch_size]

    steps_per_epoch = batches // cfg.gradient_accumulation_steps
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    log_interval = 25
    val_interval = 200

    if rank == 0:
        print(f"{EXP} Training: {total_steps} total steps, {warmup_steps} warmup steps", flush=True)

    for epoch in range(cfg.num_epochs):
        accumulated_loss = 0.0
        optimizer.zero_grad()
        for step_idx, start in enumerate(range(0, len(training_data), cfg.train_batch_size)):
            batch_list = training_data[start : start + cfg.train_batch_size]
            batch_list = materialize_missing_steering_vectors(batch_list, tokenizer, model)
            batch = construct_batch(batch_list, tokenizer, device)
            loss = train_features_batch(cfg, batch, train_module, submodule, device, dtype)
            (loss / cfg.gradient_accumulation_steps).backward()
            accumulated_loss += loss.item() / cfg.gradient_accumulation_steps

            if (step_idx + 1) % cfg.gradient_accumulation_steps == 0:
                clip_grad_norm_(train_module.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if rank == 0 and global_step % log_interval == 0:
                    ppl = math.exp(min(accumulated_loss, 20))
                    print(f"{EXP} [epoch {epoch+1} step {global_step}/{total_steps}] loss={accumulated_loss:.4f}  ppl={ppl:.2f}  lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

                if global_step % val_interval == 0 and (cfg.eval_on_start or global_step > 0):
                    if rank == 0:
                        eval_all_datasets(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step)
                    dist.barrier()

                if global_step % cfg.save_steps == 0 and global_step > 0:
                    if rank == 0:
                        model.save_pretrained(f"{cfg.save_dir}/step_{global_step}")
                    dist.barrier()

                global_step += 1
                accumulated_loss = 0.0

    if rank == 0:
        print(f"{EXP} Saving final model...")
        model.save_pretrained(f"{cfg.save_dir}/final")
        eval_all_datasets(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step)
    dist.barrier()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    model_name = "Qwen/Qwen3-4B"
    layer_percents = [75]
    train_batch_size = 32
    main_train_size = 6000
    main_test_size = 250

    dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")

    LAST_TOKEN = dict(max_window_size=1, min_window_size=1, min_end_offset=-1, max_end_offset=-1)

    classification_datasets = {
        "geometry_of_truth": {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
        "relations":         {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
        "sst2":              {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
        "md_gender":         {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
        "snli":              {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
        "ner":               {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
        "tense":             {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
        "ag_news":           {"num_train": 0,              "num_test": main_test_size, "splits": ["test"]},
        "singular_plural":   {"num_train": 0,              "num_test": main_test_size, "splits": ["test"]},
    }

    cls_loaders: list[ActDatasetLoader] = [
        ClassificationDatasetLoader(
            dataset_config=mk_cfg(
                ClassificationDatasetConfig(classification_dataset_name=ds_name, num_qa_per_sample=2, **LAST_TOKEN),
                num_train=meta["num_train"], num_test=meta["num_test"], splits=meta["splits"],
                model_name=model_name, layer_percents=layer_percents, save_acts=False, batch_size=train_batch_size,
            ),
            model_kwargs={},
        )
        for ds_name, meta in classification_datasets.items()
    ]

    latentqa_loader = LatentQADatasetLoader(
        dataset_config=mk_cfg(
            LatentQADatasetConfig(position_types=["window"], **LAST_TOKEN),
            num_train=100_000, num_test=0, splits=["train"],
            model_name=model_name, layer_percents=layer_percents, save_acts=False, batch_size=train_batch_size,
        )
    )

    # PastLens (ctx): single-activation variant — give the model 1 token's activation,
    # ask it to predict up to 50 surrounding context tokens.
    ctx_loader = PastLensDatasetLoader(
        dataset_config=mk_cfg(
            PastLensDatasetConfig(
                max_k_activations=1,    # single injection token (last-token style)
                min_k_activations=1,
                max_k_tokens=50,        # predict up to 50 surrounding tokens
            ),
            num_train=100_000, num_test=0, splits=["train"],
            model_name=model_name, layer_percents=layer_percents, save_acts=True, batch_size=train_batch_size,
        )
    )

    all_loaders = cls_loaders + [latentqa_loader, ctx_loader]

    cfg = SelfInterpTrainingConfig(
        model_name=model_name,
        hook_onto_layer=1,
        layer_percents=layer_percents,
        train_batch_size=train_batch_size,
        activation_collection_batch_size=train_batch_size * 4,
        eval_batch_size=train_batch_size * 8,
        eval_on_start=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        save_steps=5000,
        save_dir="../tests/ao/experiments/ao_cls_latentqa_ctx/checkpoints",
    )
    cfg.finalize(dataset_loaders=all_loaders)
    print(f"{EXP} Save dir: {cfg.save_dir}")

    tokenizer = load_tokenizer(model_name)
    training_data, eval_datasets = build_datasets(cfg, all_loaders)
    print(f"{EXP} Training examples: {len(training_data)}, Eval: {list(eval_datasets.keys())}")

    train_model(cfg=cfg, training_data=training_data, eval_datasets=eval_datasets,
                tokenizer=tokenizer, device=device, dtype=dtype, model_kwargs={})
