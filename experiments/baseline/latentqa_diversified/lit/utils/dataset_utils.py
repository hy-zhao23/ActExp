"""Dataset utilities for LatentQA on our diversified mix — CACHED activation variant.

This is the v2 design of the LatentQA baseline on diversified data:

- target activation is read from our pre-cached layer-27 last-token vectors
  (data/representations/finetune_diversified{_qwen3_l27}/), NOT from an online
  chat-wrapped target forward. Aligns activation source with v1-2 / AO so the
  ONLY variable across the three baselines is the decoder injection mechanism.
- decoder write path unchanged: `?`-placeholder + BASE_DIALOG + question +
  answer, with single-token layer-0 patch via forward hook.
- supervision: only the last `reflect` content (mask_all_but_last=True).

The chat-wrapped read path is dropped because the single-token patch in this
fork already voids the upstream mask_verbs purpose.
"""

import random
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from utils.finetune_dataset import FinetuneDataset


###################################
###### Tokens and formatting ######
###################################

IGNORE_IDX = -100

QWEN_NAME  = "Qwen/Qwen3-4B-Instruct-2507"
LLAMA_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# Upstream's empirical padding-shift between read placeholders and write
# tokens. Now that read is gone, we only need the write shift.
NUM_WRITE_TOKENS_TO_SHIFT = {QWEN_NAME: 3, LLAMA_NAME: 5}

# Pad-token ids per tokenizer family. Used by infra_utils.get_tokenizer to set
# tokenizer.pad_token_id explicitly (avoids the default eos==pad ambiguity).
PAD_TOKEN_IDS = {QWEN_NAME: 151643, LLAMA_NAME: 128010}

# Whether the model's chat template prepends a BOS token; relevant only for
# how write-side mask boundaries are detected (shift_start in mask_inputs).
TOKENIZER_HAS_BOS_BY_NAME = {QWEN_NAME: False, LLAMA_NAME: True}

# Chat-format token markers used by mask_inputs to find segment boundaries.
# Tuple = (system_start, user_start, assistant_start_default, assistant_start_modify).
CHAT_FORMAT_TOKENS = {
    QWEN_NAME: (
        torch.tensor([151644, 8948, 198]),   # <|im_start|>system\n
        torch.tensor([151644, 872, 198]),    # <|im_start|>user\n
        torch.tensor([151644, 77091, 198]),  # <|im_start|>assistant\n
        torch.tensor([151644, 34913, 198]),  # <|im_start|>reflect\n  (modify_chat_template)
    ),
    LLAMA_NAME: (
        torch.tensor([128006,  9125, 128007, 271]),  # <|start_header_id|>system<|end_header_id|>\n\n
        torch.tensor([128006,   882, 128007, 271]),  # <|start_header_id|>user<|end_header_id|>\n\n
        torch.tensor([128006, 78191, 128007, 271]),  # <|start_header_id|>assistant<|end_header_id|>\n\n
        torch.tensor([128006, 36013, 128007, 271]),  # <|start_header_id|>reflect<|end_header_id|>\n\n
    ),
}

# When `modify_chat_template=True`, the decoder uses this template (renames
# assistant -> reflect) so its output tokens don't collide with the target's.
DECODER_CHAT_TEMPLATES = {
    QWEN_NAME: (
        "{%- if messages[0].role == 'system' %}\n"
        "        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n"
        "    {%- endif %}\n"
        "{%- for message in messages %}\n"
        "    {%- if message.content is string %}\n"
        "        {%- set content = message.content %}\n"
        "    {%- else %}\n"
        "        {%- set content = '' %}\n"
        "    {%- endif %}\n"
        "    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n"
        "        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n"
        "    {%- elif message.role == \"assistant\" %}\n"
        "        {{- '<|im_start|>' + 'reflect' + '\\n' + content }}\n"
        "        {{- '<|im_end|>\\n' }}\n"
        "    {%- endif %}\n"
        "{%- endfor %}\n"
        "{%- if add_generation_prompt %}\n"
        "    {{- '<|im_start|>reflect\\n' }}\n"
        "{%- endif %}"
    ),
    LLAMA_NAME: (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% set role = message['role'] %}"
        "{% if role == 'assistant' %}{% set role = 'reflect' %}{% endif %}"
        "{% set content = '<|start_header_id|>' + role + '<|end_header_id|>\n\n' "
        "+ message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
        "{{ content }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>reflect<|end_header_id|>\n\n' }}"
        "{% endif %}"
    ),
}

# Prepended dialog before the QA pair (gives the decoder a stable role prefix).
BASE_DIALOG = [
    {"role": "assistant", "content": "Sure, I've analyzed the assistant."}
]


###############################
###### Write-side masking #####
###############################


def mask_inputs(
    input_ids,
    tokenizer_name,
    shift_start=False,
    mask_all_but_last=True,
    modify_chat_template=False,
):
    """Build the loss-mask for the write sequence.

    Default: `mask_all_but_last=True` — only the final `reflect` content
    (= the answer) is supervised.
    """
    sys_tokens, start_tokens, end_tokens_default, end_tokens_modify = CHAT_FORMAT_TOKENS[
        tokenizer_name
    ]
    end_tokens = end_tokens_modify if modify_chat_template else end_tokens_default
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for b in range(batch_size):
        start_idx, end_idx = [], []
        for i in range(seq_len):
            if torch.equal(input_ids[b][i : i + len(start_tokens)], start_tokens):
                start_idx.append(i)
            if torch.equal(input_ids[b][i : i + len(end_tokens)], end_tokens):
                end_idx.append(i)

        if len(start_idx) != len(end_idx):
            # Improperly formatted; mask everything to skip the row.
            mask[b][:] = True
            continue
        if mask_all_but_last:
            # Mask everything up to and including the last reflect header,
            # leaving only the last assistant/reflect content in the loss.
            mask[b][: end_idx[-1] + len(end_tokens)] = True
        else:
            # Upstream default: mask each (user-turn, next-reflect-header) span
            # but leave intervening reflect content in the loss. Kept only for
            # backward compatibility / debugging.
            for i, (start, end) in enumerate(zip(start_idx, end_idx)):
                if shift_start and i == 0:
                    mask[b][start - 1 : end + len(end_tokens)] = True
                else:
                    mask[b][start : end + len(end_tokens)] = True
    return mask


def lqa_tokenize_cached(
    batch,
    tokenizer,
    name=None,
    generate=False,
    mask_all_but_last=True,
    modify_chat_template=True,
):
    """Tokenize the write sequence for the cached-activation LatentQA path.

    Input batch items: {vector: (d,) bf16, dialog: list[dict]}.
    The write sequence is: `?`-placeholder + BASE_DIALOG + question + answer
    (dialog already has BASE_DIALOG prepended in __getitem__).

    Returns a dict with:
        vector          : (B, d) bfloat16 — the cached read activation
        tokenized_write : padded BatchEncoding with `labels` (loss mask applied)
        write_lengths   : (B,) — content lengths (minus padding-shift constant)
    """
    name = tokenizer.name_or_path if name is None else name
    TOKENIZER_HAS_BOS = TOKENIZER_HAS_BOS_BY_NAME[name]

    # Build the write text: `?` placeholder gives the decoder exactly ONE
    # position to receive the patched activation.
    queries = []
    for item in batch:
        query = [{"role": "user", "content": "? "}]  # 1 placeholder token
        query += item["dialog"]
        queries.append(
            tokenizer.apply_chat_template(
                query,
                tokenize=False,
                add_generation_prompt=generate,
                chat_template=(DECODER_CHAT_TEMPLATES[name] if modify_chat_template else None),
            )
        )
    tokenized_write = tokenizer(
        queries,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )

    write_lengths = torch.sum(tokenized_write.attention_mask, dim=1)
    write_lengths = write_lengths - NUM_WRITE_TOKENS_TO_SHIFT[name]

    if not generate:
        user_inputs_mask = mask_inputs(
            tokenized_write.input_ids,
            name,
            shift_start=TOKENIZER_HAS_BOS,
            mask_all_but_last=mask_all_but_last,
            modify_chat_template=modify_chat_template,
        )
        assert tokenizer.padding_side == "left"
        tokenized_write["labels"] = tokenized_write.input_ids.clone()
        mask = (tokenized_write.attention_mask == 0) | user_inputs_mask
        tokenized_write["labels"][mask] = IGNORE_IDX

    vectors = torch.stack([item["vector"] for item in batch])  # (B, d)

    return {
        "vector":          vectors,
        "tokenized_write": tokenized_write,
        "write_lengths":   write_lengths,
    }


###########################
####### Dataset class #####
###########################


class CachedLatentQADataset(Dataset):
    """Thin wrapper around utils.finetune_dataset.FinetuneDataset.

    Reuses ours' physical train/val/test split + (vector, text, question,
    answer) packaging. Emits LatentQA-style write-side items:
        vector : (d,) bf16     — the cached last-token layer-27 activation
        dialog : list[dict]    — BASE_DIALOG + [user: q, assistant: a]
    """

    def __init__(self, base: FinetuneDataset):
        self.base = base
        self.hidden_dim = base.hidden_dim
        self.source_model = base.source_model

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        rec = self.base[idx]
        qa_dialog = [
            {"role": "user",      "content": rec["question"]},
            {"role": "assistant", "content": rec["label"]},
        ]
        return {
            "vector": rec["vector"],
            "dialog": BASE_DIALOG + qa_dialog,
        }


class DataCollatorForLatentQA:
    def __init__(
        self,
        tokenizer,
        generate=False,
        mask_all_but_last=True,
        modify_chat_template=True,
    ):
        self.tokenizer = tokenizer
        self.generate = generate
        self.mask_all_but_last = mask_all_but_last
        self.modify_chat_template = modify_chat_template

    def __call__(self, batch):
        return lqa_tokenize_cached(
            batch,
            self.tokenizer,
            generate=self.generate,
            mask_all_but_last=self.mask_all_but_last,
            modify_chat_template=self.modify_chat_template,
        )


###############################
####### Public entry points ###
###############################


def get_dataloaders(train_config, tokenizer):
    """Build train / eval DataLoaders backed by FinetuneDataset cache."""
    base_train = FinetuneDataset(
        split="train",
        subdir=train_config.cache_subdir,
        raw_subdir=train_config.raw_subdir,
        qa_subdir=train_config.qa_subdir,
        train_subsample_ratio=(
            train_config.train_subsample_ratio
            if train_config.train_subsample_ratio > 0 else None
        ),
        sample_seed=train_config.seed,
    )
    dataset_train = CachedLatentQADataset(base_train)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        seed=train_config.seed,
        drop_last=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        sampler=train_sampler,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        collate_fn=DataCollatorForLatentQA(
            tokenizer,
            mask_all_but_last=True,
            modify_chat_template=train_config.modify_chat_template,
        ),
        drop_last=True,
    )

    eval_dataloader = None
    if train_config.eval_ppl:
        base_eval = FinetuneDataset(
            split="val",
            subdir=train_config.cache_subdir,
            raw_subdir=train_config.raw_subdir,
            qa_subdir=train_config.qa_subdir,
        )
        dataset_eval = CachedLatentQADataset(base_eval)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_eval,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=True,
        )
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=train_config.batch_size_training,
            sampler=eval_sampler,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            collate_fn=DataCollatorForLatentQA(
                tokenizer,
                mask_all_but_last=True,
                modify_chat_template=train_config.modify_chat_template,
            ),
            drop_last=True,
        )

    return train_dataloader, eval_dataloader
