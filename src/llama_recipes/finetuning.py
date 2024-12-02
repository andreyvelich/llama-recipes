# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of
# the Llama 2 Community License Agreement.


import os
import random

import fire
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist

import functools


from configs.configs import (
    fsdp_config as FSDP_CONFIG,
    train_config as TRAIN_CONFIG,
    KubeflowLoraConfig,
)
from data.data import ConcatDataset

from utils.config_utils import (
    get_dataloader_kwargs,
    update_config,
)
from utils.train_utils import train

from peft import get_peft_model, LoraConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
)

from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


import copy
import json
from torch.utils.data import Dataset
import pandas as pd


# Path to Alpaca dataset.
DATA_PATH = "/workspace/dataset/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"

# Path to Llama 3.2-1B model.
MODEL_PATH = "/workspace/model"

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input "
        "that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def fsdp_auto_wrap_policy(transformer_layer_names):

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(
        lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn
    )
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls=set(transformer_layer_names)
    )

    auto_wrap_policy = functools.partial(
        _or_policy, policies=[lambda_policy, transformer_wrap_policy]
    )
    return auto_wrap_policy


class InstructionDataset(Dataset):
    def __init__(self, tokenizer, partition="train"):

        df = pd.read_parquet(DATA_PATH)

        # Process only 10000 samples for the demo.
        self.ann = df.to_dict(orient="records")[:10000]

        # Use 5% of the dataset for evaluation
        eval_length = int(len(self.ann) / 20)
        if partition == "train":
            self.ann = self.ann[eval_length:]
        else:
            self.ann = self.ann[:eval_length]

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask": example_mask.tolist(),
        }


def main(**kwargs):

    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
    # Set the seeds for reproducibility
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)

    dist.init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if dist.get_rank() == 0:
        print("-" * 50)
        print("Initialize PyTorch FSDP Training")
        print(
            f"Training Node. LOCAL_RANK: {local_rank}, RANK: {rank}, WORLD_SIZE: {world_size}\n\n"
        )

    torch.cuda.set_device(local_rank)

    # Load the pre-trained model and setup its configuration
    if dist.get_rank() == 0:
        print("-" * 50)
        print(
            f"Load meta-llama/Llama-3.2-1B-Instruct model and tokenizer from the {MODEL_PATH}"
        )

    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # TODO (andreyvelich): Get Lora Config from the envs.
    env_lora_config = os.getenv("LORA_CONFIG", None)
    env_lora_config = KubeflowLoraConfig(**json.loads(env_lora_config))
    reference_lora_config = KubeflowLoraConfig()

    for key, val in env_lora_config.__dict__.items():
        old_attr = getattr(reference_lora_config, key, None)
        if old_attr is not None and val is None:
            val = old_attr
        setattr(env_lora_config, key, val)

    peft_config = LoraConfig()
    # TODO: Only set rank for the demo.
    peft_config.r = env_lora_config.r

    # TODO: The default LoRA values.
    peft_config.lora_alpha = 32
    peft_config.lora_dropout = 0.05
    peft_config.target_modules = ["q_proj", "v_proj"]

    model = get_peft_model(model, peft_config)

    if dist.get_rank() == 0:
        model.print_trainable_parameters()
        print(f"Using the LoRA config for the PEFT: {env_lora_config}")
        print("\n\n")

    # Read dataset.
    dataset_train = InstructionDataset(
        tokenizer,
        partition="train",
    )
    dataset_val = InstructionDataset(
        tokenizer,
        partition="val",
    )

    if dist.get_rank() == 0:
        print("-" * 50)
        print(f"Load Training and Validation dataset from the {DATA_PATH}")
        print(f"Train dataset length: {len(dataset_train.ann)}")
        print(f"Val dataset length: {len(dataset_val.ann)}\n\n")

    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,  # Gradient communication precision.
        buffer_dtype=torch.float16,  # Buffer precision.
    )

    device_id = torch.cuda.current_device()
    model = FSDP(
        model,
        auto_wrap_policy=fsdp_auto_wrap_policy([LlamaDecoderLayer]),
        cpu_offload=(
            CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None
        ),
        mixed_precision=mixed_precision_policy,
        sharding_strategy=fsdp_config.sharding_strategy,
        device_id=device_id,
        limit_all_gathers=True,
        sync_module_states=train_config.low_cpu_fsdp,
        param_init_fn=(
            (lambda module: module.to_empty(device=torch.device("cuda"), recurse=False))
            if train_config.low_cpu_fsdp and rank != 0
            else None
        ),
    )

    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(
            dataset_train, chunk_size=train_config.context_length
        )

    train_dl_kwargs = get_dataloader_kwargs(
        train_config, dataset_train, tokenizer, "train"
    )

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(
                dataset_val, chunk_size=train_config.context_length
            )

        val_dl_kwargs = get_dataloader_kwargs(
            train_config, dataset_val, tokenizer, "val"
        )

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )

        if len(eval_dataloader) == 0:
            raise ValueError(
                "The eval set size is too small for dataloader to load even one batch. "
                "Please increase the size of eval set. ({len(eval_dataloader)=})"
            )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        local_rank,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
