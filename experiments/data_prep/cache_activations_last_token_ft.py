"""
Pre-cache activation vectors for the last-token oracle fine-tuning setting.

Differences from baseline cache_activations.py:
  - Classification : only single-token variant; end_offset fixed to -1 (true last token)
  - LatentQA       : position_types=["window"], window_size=1, end_offset fixed to -1
  - PastLens       : only single variant (max_k_activations=1); act position is already
                     the last token of its context by construction

The config hash differs from baseline files → new .pt files are written
alongside existing ones without overwriting them.

Usage:
  python experiments/data_prep/cache_activations_last_token_ft.py --model-name Qwen/Qwen3-4B
  # or via SLURM: sbatch run/oracle/11_cache_last_token.sh
"""

import argparse
import gc
import os
import sys

# nl_probes lives inside the baseline activation_oracles submodule
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../baseline/activation_oracles"))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader, DatasetLoaderConfig
from nl_probes.dataset_classes.classification import ClassificationDatasetConfig, ClassificationDatasetLoader
from nl_probes.dataset_classes.latentqa_dataset import LatentQADatasetConfig, LatentQADatasetLoader
from nl_probes.dataset_classes.past_lens_dataset import PastLensDatasetConfig, PastLensDatasetLoader
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import load_model
from nl_probes.utils.dataset_utils import TrainingDataPoint


# ---------------------------------------------------------------------------
# Loader construction
# ---------------------------------------------------------------------------

def _mk_cfg(
    custom_params,
    *,
    num_train: int,
    num_test: int,
    splits: list[str],
    model_name: str,
    layer_percents: list[int],
    batch_size: int,
    save_acts: bool = False,
) -> DatasetLoaderConfig:
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


def build_loaders(
    model_name: str,
    layer_percents: list[int],
    classification_datasets: dict,
    batch_size: int,
) -> list[ActDatasetLoader]:
    """
    Last-token setting loaders:
      - Classification : single-token only, end_offset=-1/-1
      - LatentQA       : window=1, end_offset=-1/-1
      - PastLens       : single variant only (max_k_activations=1)
    """
    num_datapoints = 100_000
    loaders: list[ActDatasetLoader] = []

    # ── LatentQA ──────────────────────────────────────────────────────────────
    loaders.append(
        LatentQADatasetLoader(
            dataset_config=_mk_cfg(
                LatentQADatasetConfig(
                    position_types=["window"],
                    max_window_size=1,
                    min_window_size=1,
                    min_end_offset=-1,
                    max_end_offset=-1,
                ),
                num_train=100_000,
                num_test=0,
                splits=["train"],
                model_name=model_name,
                layer_percents=layer_percents,
                batch_size=batch_size,
            )
        )
    )

    # ── PastLens single (last-token by construction when k_acts=1) ────────────
    loaders.append(
        PastLensDatasetLoader(
            dataset_config=_mk_cfg(
                PastLensDatasetConfig(
                    max_k_activations=1,
                    max_k_tokens=50,
                ),
                num_train=num_datapoints,
                num_test=0,
                splits=["train"],
                model_name=model_name,
                layer_percents=layer_percents,
                batch_size=batch_size,
            )
        )
    )

    # ── Classification — single-token, last token ──────────────────────────────
    for ds_name, meta in classification_datasets.items():
        bs = meta.get("batch_size", batch_size)
        loaders.append(
            ClassificationDatasetLoader(
                dataset_config=_mk_cfg(
                    ClassificationDatasetConfig(
                        classification_dataset_name=ds_name,
                        max_window_size=1,
                        min_window_size=1,
                        min_end_offset=-1,
                        max_end_offset=-1,
                        num_qa_per_sample=2,
                    ),
                    num_train=meta["num_train"],
                    num_test=meta.get("num_test", 0),
                    splits=meta["splits"],
                    model_name=model_name,
                    layer_percents=layer_percents,
                    batch_size=bs,
                )
            )
        )

    return loaders


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------

@torch.no_grad()
def materialize_lazy_datapoints(
    data: list[TrainingDataPoint],
    model: AutoModelForCausalLM,
    batch_size: int,
    device: torch.device,
) -> list[TrainingDataPoint]:
    to_fill = [(i, dp) for i, dp in enumerate(data) if dp.steering_vectors is None]
    if not to_fill:
        return data

    print(f"  Materializing {len(to_fill)}/{len(data)} lazy items (batch_size={batch_size})")
    result = list(data)
    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0

    for start in tqdm(range(0, len(to_fill), batch_size), desc="  Caching activations"):
        batch = to_fill[start : start + batch_size]

        layers_needed = sorted({dp.layer for _, dp in batch})
        submodules = {layer: get_hf_submodule(model, layer) for layer in layers_needed}

        assert all(dp.context_input_ids is not None and dp.context_positions is not None for _, dp in batch)
        contexts = [list(dp.context_input_ids) for _, dp in batch]  # type: ignore[arg-type]
        positions_per_item = [list(dp.context_positions) for _, dp in batch]  # type: ignore[arg-type]
        max_len = max(len(c) for c in contexts)

        input_ids_list = []
        attn_masks_list = []
        left_offsets = []
        for c in contexts:
            pad = max_len - len(c)
            input_ids_list.append(torch.tensor([pad_id] * pad + c, dtype=torch.long, device=device))
            attn_masks_list.append(torch.tensor([False] * pad + [True] * len(c), dtype=torch.bool, device=device))
            left_offsets.append(pad)

        inputs_BL = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attn_masks_list),
        }

        acts_by_layer = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=inputs_BL,
            min_offset=None,
            max_offset=None,
        )

        for b, (orig_idx, dp) in enumerate(batch):
            idxs = [p + left_offsets[b] for p in positions_per_item[b]]
            vectors = acts_by_layer[dp.layer][b, idxs, :].detach().cpu().contiguous()
            dp_new = dp.model_copy(deep=True)
            dp_new.steering_vectors = vectors
            result[orig_idx] = dp_new

        del acts_by_layer, inputs_BL, input_ids_list, attn_masks_list
        torch.cuda.empty_cache()

    return result


def _done_marker(loader: ActDatasetLoader, split: str) -> str:
    filename = loader.get_dataset_filename(split)  # type: ignore[arg-type]
    return os.path.join(loader.dataset_config.dataset_folder, filename + ".cache_done")


def cache_loader(
    loader: ActDatasetLoader,
    model: AutoModelForCausalLM,
    device: torch.device,
    batch_size: int,
) -> None:
    for split in loader.dataset_config.splits:
        marker = _done_marker(loader, split)
        print(f"\n[cache] {loader.dataset_config.dataset_name!r}  split={split!r}")

        if os.path.exists(marker):
            print(f"  Already done (marker found), skipping.")
            continue

        data = loader.load_dataset(split)  # type: ignore[arg-type]

        lazy_count = sum(1 for dp in data if dp.steering_vectors is None)
        if lazy_count == 0:
            print(f"  Already fully materialized ({len(data)} items); writing marker.")
            open(marker, "w").close()
            continue

        data = materialize_lazy_datapoints(data, model, batch_size, device)
        loader.save_dataset(data, split)  # type: ignore[arg-type]
        open(marker, "w").close()
        print(f"  Saved {len(data)} materialized items; marker written.")

    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-cache activations — last-token fine-tuning setting")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--latentqa-batch-size", type=int, default=512)
    args = parser.parse_args()

    rank = args.rank if args.rank is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    world_size = args.world_size if args.world_size is not None else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    main_train_size = 6000
    main_test_size = 250
    classification_datasets: dict = {
        "geometry_of_truth": {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
        "relations":         {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
        "sst2":              {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
        "md_gender":         {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
        "snli":              {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
        "ag_news":           {"num_train": main_train_size, "num_test": main_test_size, "splits": ["test"]},
        "ner":               {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
        "tense":             {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
        "language_identification": {
            "num_train": main_train_size, "num_test": main_test_size,
            "splits": ["test"],
            "batch_size": 4,
        },
        "singular_plural": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
    }
    layer_percents = [25, 50, 75]

    # cd into activation_oracles so relative paths in loaders (e.g. sft_training_data/) resolve correctly
    ao_dir = os.path.join(os.path.dirname(__file__), "../baseline/activation_oracles")
    os.chdir(os.path.realpath(ao_dir))

    all_loaders = build_loaders(
        model_name=args.model_name,
        layer_percents=layer_percents,
        classification_datasets=classification_datasets,
        batch_size=16,
    )

    my_loaders = [ldr for i, ldr in enumerate(all_loaders) if i % world_size == rank]

    print(
        f"[rank {rank}/{world_size}] {len(my_loaders)}/{len(all_loaders)} loaders"
        f" | model={args.model_name} | batch_size={args.batch_size}"
        f" | latentqa_batch_size={args.latentqa_batch_size}"
    )

    model = load_model(args.model_name, torch.bfloat16, device_map={"": device})
    model.eval()

    for loader in my_loaders:
        bs = args.latentqa_batch_size if isinstance(loader, LatentQADatasetLoader) else args.batch_size
        cache_loader(loader, model, device, bs)

    print(f"\n[rank {rank}] All done.")
