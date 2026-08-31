import os
import json
from datetime import datetime
import logging
from glob import glob
from dataclasses import is_dataclass
from collections import OrderedDict
from functools import partial
from copy import deepcopy

from accelerate import init_empty_weights
from transformers import AutoConfig

from peft import get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)
from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder

from lit.utils.dataset_utils import PAD_TOKEN_IDS


###################
###### Utils ######
###################


def update_config(config, **kwargs):
    def update_nested(obj, key, value):
        if hasattr(obj, key):
            if is_dataclass(getattr(obj, key)):
                update_config(
                    getattr(obj, key),
                    **{k: v for k, v in kwargs.items() if k.startswith(f"{key}.")},
                )
            else:
                setattr(obj, key, value)
        elif hasattr(obj, "peft_config") and hasattr(obj.peft_config, key):
            # This line handles the case of --lora_alpha 64
            setattr(obj.peft_config, key, value)
        elif "." in key:
            parent, child = key.split(".", 1)
            if hasattr(obj, parent):
                update_nested(getattr(obj, parent), child, value)
        else:
            if type(obj).__name__ not in [
                "steering_config",
                "fsdp_config",
            ]:
                print(f"Warning: {type(obj).__name__} does not accept parameter: {key}")

    for k, v in kwargs.items():
        update_nested(config, k, v)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    if ema_model is None:
        return
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if param.requires_grad:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


##########################
###### Logging code ######
##########################


def save_model(decoder_model, ema_model, tokenizer, args, epoch, steps, logger, rank):
    if rank == 0:
        logger.info(f"Saving decoder model...")
        output_dir = (
            args.checkpoint_dir
            + f"/epoch{epoch}-steps{steps}"
            + f"-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        ema_output_dir = (
            args.checkpoint_dir
            + f"/ema-epoch{epoch}-steps{steps}"
            + f"-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
    else:
        output_dir, ema_output_dir = None, None
    for dir, model, name in [
        (output_dir, decoder_model, "model"),
        (ema_output_dir, ema_model, "ema model"),
    ]:
        if model is None:
            continue
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(model, options=options)
        if rank == 0:
            model.save_pretrained(dir, state_dict=state_dict)
            tokenizer.save_pretrained(dir)
            logger.info(f"{name} is saved in {dir} directory")


def create_logger(logging_dir, rank):
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    logger.propagate = False

    if rank == 0:
        logger.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            fmt="[\033[34m%(asctime)s\033[0m] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_formatter = logging.Formatter(
            fmt="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(f"{logging_dir}/log.txt")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    else:
        logger.setLevel(logging.ERROR)
        logger.addHandler(logging.NullHandler())
    return logger


def get_logger(args, rank):
    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.output_dir}/*"))
        experiment_dir = f"{args.output_dir}/{experiment_index:03d}"
        args.checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, rank)
        logger.info(f"Experiment directory created at {experiment_dir}")
        # Dump args into the experiment folder:
        with open(f"{experiment_dir}/exp_args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    else:
        logger = create_logger(None, rank)
    return logger


################################
###### Model loading code ######
################################


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", add_eos_token=True
    )
    tokenizer.pad_token_id = PAD_TOKEN_IDS[model_name]
    if "distill-qwen" in model_name.lower():
        tokenizer.add_tokens(["<|reserved_special_token_8|>"])
    return tokenizer


def fsdp_auto_wrap_policy(model, transformer_layer_name):
    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(
            PrefixEncoder,
            PromptEncoder,
            PromptEmbedding,
            transformer_layer_name,
        ),
    )
    auto_wrap_policy = partial(
        _or_policy, policies=[lambda_policy, transformer_wrap_policy]
    )
    return auto_wrap_policy


def get_model(
    model_name,
    tokenizer,
    peft_config=None,
    load_peft_checkpoint=None,
    fsdp_args=None,
    device=None,
    rank=None,
    distributed_training=False,
):
    if fsdp_args is not None and fsdp_args.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        if rank == 0:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                use_cache=None,
                torch_dtype=torch.bfloat16,
            )
        else:
            config = AutoConfig.from_pretrained(model_name)   # tiny
            with init_empty_weights():                        # everything on meta
                model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            #use_cache=None,
            device_map="auto" if device == "auto" else None,
        )
    
    # --- operations that read the weights -------------------------------
    model.resize_token_embeddings(len(tokenizer))
    for p in model.parameters():
        p.requires_grad = False

    # Load PEFT
    assert peft_config is None or load_peft_checkpoint is None
    use_peft = peft_config is not None or load_peft_checkpoint is not None
    if peft_config is not None:
        model = get_peft_model(model, peft_config)
    elif load_peft_checkpoint is not None:
        model = PeftModel.from_pretrained(model, load_peft_checkpoint)

    # Distribute models
    if fsdp_args is None:
        if device is not None and device != "auto":
            model = model.to(device)
        if distributed_training:
            # `device` is the local device index (set in train.py from LOCAL_RANK).
            # Upstream passed `rank` here, which only works for single-node DDP.
            model = DDP(model, device_ids=[device])
        return model
    else:
        hsdp_device_mesh = None
        if (
            fsdp_args.hsdp
            and fsdp_args.sharding_strategy == ShardingStrategy.HYBRID_SHARD
        ):
            hsdp_device_mesh = hsdp_device_mesh(
                replica_group_size=fsdp_args.replica_group_size,
                sharding_group_size=fsdp_args.sharding_group_size,
            )
            print("HSDP device mesh is ready")

        if "llama" in model_name:
            DECODER_LAYER = LlamaDecoderLayer
        elif "mistral" in model_name:
            DECODER_LAYER = MistralDecoderLayer
        elif "qwen2" in model_name:
            DECODER_LAYER = Qwen2DecoderLayer
            
        wrapping_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                DECODER_LAYER,
            },
        )
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, DECODER_LAYER)
        device_id = torch.cuda.current_device()

        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy if use_peft else wrapping_policy,
            cpu_offload=(
                CPUOffload(offload_params=True) if fsdp_args.fsdp_cpu_offload else None
            ),
            mixed_precision=None,
            sharding_strategy=fsdp_args.sharding_strategy,
            device_mesh=hsdp_device_mesh,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=fsdp_args.low_cpu_fsdp,
            param_init_fn=lambda module: (
                module.to_empty(device=torch.device("cuda"), recurse=False)
                if fsdp_args.low_cpu_fsdp and rank != 0
                else None
            ),
        )
        if fsdp_args.fsdp_activation_checkpointing:
            non_reentrant_wrapper = partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.REENTRANT,
            )
            check_fn = lambda submodule: isinstance(submodule, DECODER_LAYER)
            apply_activation_checkpointing(
                model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
            )
        return model


def get_modules(
    target_model,
    decoder_model,
    min_layer_to_read=15,
    max_layer_to_read=16,
    num_layers_to_read=1,
    layer_to_write=0,
    module_setup="read-vary_write-fixed_n-fixed",
    **kwargs,
):
    # If in eval mode, target_model has LORA layers attached to it
    try:
        eval("target_model.model.layers")
        target_model_str = "target_model.model"
    except:
        try:
            eval("target_model.model.model.layers")
            target_model_str = "target_model.model.model"
        except:
            try:
                eval("target_model.module.model.model.layers")
                target_model_str = "target_model.module.model.model"
            except:
                # Gemma-series models use model.language_model bc of multimodality
                try:
                    eval("target_model.language_model.model.layers")
                    target_model_str = "target_model.language_model.model"
                except:
                    target_model_str = "target_model.module.language_model.model"
    try:
        eval("decoder_model.model.layers")
        decoder_model_str = "decoder_model.model"
    except:
        try:
            eval("decoder_model.model.model.layers")
            decoder_model_str = "decoder_model.model.model"
        except:
            try:
                eval("decoder_model.module.model.model.layers")
                decoder_model_str = "decoder_model.module.model.model"
            except:
                # Gemma-series models use model.language_model bc of multimodality
                try:
                    eval("decoder_model.language_model.model.layers")
                    decoder_model_str = "decoder_model.language_model.model"
                except:
                    decoder_model_str = "decoder_model.module.language_model.model"

    # List[List[Module]]
    module_read, module_write = [], []
    for i in range(min_layer_to_read, max_layer_to_read):
        module_read_i, module_write_i = [], []
        if module_setup == "read-vary_write-vary_n-fixed":
            for j in range(i, i + num_layers_to_read):
                module_read_i.append(eval(f"{target_model_str}.layers[{j}]"))
                module_write_i.append(eval(f"{decoder_model_str}.layers[{j}]"))
        elif module_setup == "read-vary_write-vary_n-vary":
            for j in range(i):
                module_read_i.append(eval(f"{target_model_str}.layers[{j}]"))
                module_write_i.append(eval(f"{decoder_model_str}.layers[{j}]"))
        elif module_setup == "read-vary_write-fixed_n-fixed":
            for j in range(i, i + num_layers_to_read):
                module_read_i.append(eval(f"{target_model_str}.layers[{j}]"))
            for j in range(layer_to_write, layer_to_write + num_layers_to_read):
                module_write_i.append(eval(f"{decoder_model_str}.layers[{j}]"))
        else:
            raise NotImplementedError
        module_read.append(module_read_i)
        module_write.append(module_write_i)
    return module_read, module_write


def get_ema(model, decay, device):
    ema = None
    if decay < 1:
        # Create an EMA of the model for use after training
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)
    # Ensure EMA is initialized with synced weights
    update_ema(ema, model, decay=0)
    return ema
