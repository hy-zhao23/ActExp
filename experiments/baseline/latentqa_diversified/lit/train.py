import os
from dataclasses import fields
import random
import fire
from tqdm import tqdm

import numpy as np
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True"  # prevent memory fragmentation
)

from lit.configs.train_config import train_config
from lit.configs.peft_config import lora_config
from lit.utils.dataset_utils import get_dataloaders
from lit.utils.infra_utils import (
    get_logger,
    save_model,
    get_ema,
    update_ema,
    update_config,
    get_tokenizer,
    get_model,
)
from lit.utils.activation_utils import latent_qa


def _resolve_write_layer(decoder_model, layer_to_write: int):
    """Return decoder_model.layers[layer_to_write], robust to DDP+PEFT wrapping."""
    candidates = (
        "decoder_model.module.base_model.model.model.layers",  # DDP(PEFT(LM))
        "decoder_model.module.model.model.layers",             # DDP(LM) legacy path
        "decoder_model.base_model.model.model.layers",         # PEFT(LM) no DDP
        "decoder_model.model.model.layers",
        "decoder_model.model.layers",
    )
    for path in candidates:
        try:
            return eval(path)[layer_to_write]
        except (AttributeError, TypeError, IndexError):
            continue
    raise RuntimeError(
        f"Could not locate decoder layers under any of: {candidates}"
    )


def main(**kwargs):
    # Get args and setup DDP
    dist.init_process_group("nccl")
    assert torch.cuda.is_available()
    args = train_config()
    update_config(args, **kwargs)
    fsdp_args = None
    if args.use_fsdp:
        from lit.configs.fsdp_config import fsdp_config

        fsdp_args = fsdp_config()
        update_config(fsdp_args, **kwargs)

    rank = dist.get_rank()
    # Multi-node DDP with 1 GPU/node: use LOCAL_RANK from torchrun for device
    # selection. Upstream's `rank % device_count()` happens to work for the
    # single-node 8×A100 case but breaks when global_rank != local index.
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = local_rank
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    seed = args.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    logger = get_logger(args, rank)
    # Load tokenizer and datasets
    tokenizer = get_tokenizer(args.target_model_name)
    train_dataloader, eval_dataloader = get_dataloaders(args, tokenizer)

    # Load the decoder only — target activation comes from disk cache
    # (data/representations/<cache_subdir>/), not an online target forward.
    # This is the v2 cached-activation pipeline.
    lora_params = {
        k.name: getattr(lora_config(), k.name) for k in fields(lora_config())
    }
    peft_config = LoraConfig(**lora_params)
    decoder_model = get_model(
        args.target_model_name,
        tokenizer,
        peft_config=peft_config,
        fsdp_args=fsdp_args,
        device=device,
        rank=rank,
        distributed_training=True,
    )
    torch.cuda.empty_cache()
    if rank == 0:
        decoder_model.module.print_trainable_parameters()
    module_write = [_resolve_write_layer(decoder_model, args.layer_to_write)]
    ema = get_ema(decoder_model.module, decay=args.ema_decay, device=device)

    # Initialize the optimizer and learning rate scheduler
    optimizer = optim.AdamW(
        decoder_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    training_steps = len(train_dataloader) * args.num_epochs
    logger.info(f"Training steps: {training_steps}")
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=training_steps,
    )

    # Start the training.
    # `train_steps` counts micro-batches (1 per inner-loop iteration);
    # `opt_steps` counts optimizer.step() calls. Eval/save trigger on opt_steps
    # so the cadence is independent of gradient_accumulation_steps.
    train_steps = 0
    opt_steps = 0

    # Early-stop bookkeeping (added in this fork; mirrors our other pipelines)
    best_eval_loss = float("inf")
    no_improve = 0
    should_stop = False

    # Track raw per-micro-step CE so train/loss log == real CE (not CE / gas).
    # Reset at every opt_step boundary; log average over the gas micro-steps.
    accum_loss = 0.0

    for epoch in range(args.num_epochs):
        if should_stop:
            break
        decoder_model.train()
        total_length = len(train_dataloader) // args.gradient_accumulation_steps
        pbar = tqdm(
            colour="blue",
            desc=f"Training Epoch: {epoch+1}",
            total=total_length,
            dynamic_ncols=True,
        )
        for step, batch in enumerate(train_dataloader):
            # `batch` from CachedLatentQADataset: vector tensor + tokenized_write
            # dict (also tensor leaves) + write_lengths tensor. Move all tensor
            # leaves to device; tokenized_write is a BatchEncoding (dict-like).
            batch["vector"] = batch["vector"].to(device)
            batch["write_lengths"] = batch["write_lengths"].to(device)
            for k in batch["tokenized_write"]:
                batch["tokenized_write"][k] = batch["tokenized_write"][k].to(device)

            train_steps += 1
            outputs = latent_qa(
                batch,
                decoder_model,
                module_write,
                tokenizer,
            )
            # Accumulate the RAW per-micro-step CE for logging; the
            # grad-accum scaling lives ONLY in the backward call so the
            # `loss` variable stays untouched for downstream logging.
            accum_loss += outputs.loss.detach().float().item()
            (outputs.loss / args.gradient_accumulation_steps).backward()
            if train_steps % args.gradient_accumulation_steps == 0:
                if (
                    args.gradient_clipping
                    and args.gradient_clipping_threshold > 0.0
                ):
                    torch.nn.utils.clip_grad_norm_(
                        decoder_model.parameters(),
                        args.gradient_clipping_threshold,
                    )
                optimizer.step()
                optimizer.zero_grad()
                update_ema(ema, decoder_model.module, decay=args.ema_decay)
                opt_steps += 1
                pbar.update(1)

                # Log on opt-step boundary (cadence matches v1-2 / AO).
                avg_loss = accum_loss / args.gradient_accumulation_steps
                accum_loss = 0.0
                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{args.num_epochs}, "
                    f"opt_step {opt_steps} (loss: {avg_loss:.4f})"
                )

            if (
                args.eval_ppl
                and opt_steps > 0
                and opt_steps % args.eval_every_n_steps == 0
                and train_steps % args.gradient_accumulation_steps == 0
            ):
                assert eval_dataloader is not None
                decoder_model.eval()
                total_loss = torch.zeros(1, device=f"cuda:{device}")
                pbar_eval = tqdm(
                    colour="green",
                    desc=f"Evaluating Epoch: {epoch+1}",
                    total=len(eval_dataloader),
                    dynamic_ncols=True,
                )
                for _, batch_e in enumerate(eval_dataloader):
                    batch_e["vector"] = batch_e["vector"].to(device)
                    batch_e["write_lengths"] = batch_e["write_lengths"].to(device)
                    for k in batch_e["tokenized_write"]:
                        batch_e["tokenized_write"][k] = batch_e["tokenized_write"][k].to(device)
                    outputs = latent_qa(
                        batch_e,
                        decoder_model,
                        module_write,
                        tokenizer,
                        no_grad=True,
                    )
                    total_loss += outputs.loss.detach().float()
                    pbar_eval.update(1)
                pbar_eval.close()

                # All-reduce so every rank has the same eval loss → identical
                # early-stop decisions without needing a broadcast.
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                eval_loss = (total_loss / (len(eval_dataloader) * dist.get_world_size())).item()

                eval_ppl = float(np.exp(eval_loss))
                if rank == 0:
                    logger.info(
                        f"[eval] opt_step={opt_steps} micro_step={train_steps} "
                        f"eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f} best={best_eval_loss:.4f}"
                    )

                # Early-stop check (every rank in lockstep)
                if args.early_stop_enabled:
                    if eval_loss < best_eval_loss - args.early_stop_min_delta:
                        best_eval_loss = eval_loss
                        no_improve = 0
                    else:
                        no_improve += 1
                        if rank == 0:
                            logger.info(
                                f"[early-stop] no improve {no_improve}/{args.early_stop_patience}"
                            )
                        if no_improve >= args.early_stop_patience:
                            if rank == 0:
                                logger.info(f"[early-stop] triggered at opt_step {opt_steps}")
                            should_stop = True

                decoder_model.train()
                if should_stop:
                    break

            if (
                opt_steps > 0
                and opt_steps % args.save_every_n_steps == 0
                and train_steps % args.gradient_accumulation_steps == 0
            ):
                save_model(
                    decoder_model if args.use_fsdp else decoder_model.module,
                    ema,
                    tokenizer,
                    args,
                    epoch,
                    train_steps,
                    logger,
                    rank,
                )

        # End of epoch
        scheduler.step()
        pbar.close()

        if args.save_model:
            save_model(
                decoder_model if args.use_fsdp else decoder_model.module,
                ema,
                tokenizer,
                args,
                epoch,
                train_steps,
                logger,
                rank,
            )
            dist.barrier()

    dist.destroy_process_group()
    logger.info("Training completed!")


if __name__ == "__main__":
    fire.Fire(main)
