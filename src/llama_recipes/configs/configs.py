# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the
# Llama 2 Community License Agreement.

from dataclasses import dataclass
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


# TODO (andreyvelich): This is Lora Config from the Kubeflow Training SDK.
@dataclass
class KubeflowLoraConfig:
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class train_config:
    model_name: str = "PATH/to/Model"
    tokenizer_name: str = None
    # shards model parameters, optimizer states and gradients across DDP ranks
    enable_fsdp: bool = False
    # saves cpu memory by loading pretrained model on rank0 only
    low_cpu_fsdp: bool = False
    run_validation: bool = True
    batch_size_training: int = 4
    batching_strategy: str = "packing"  # alternative: padding
    context_length: int = 4096
    gradient_accumulation_steps: int = 1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int = 3
    max_train_step: int = 0
    max_eval_step: int = 0
    num_workers_dataloader: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    # multiplicatively decay the learning rate by gamma after each epoch
    gamma: float = 0.85
    seed: int = 42
    use_fp16: bool = False
    mixed_precision: bool = True
    val_batch_size: int = 1
    dataset = "samsum_dataset"
    # None, llama_adapter (Caution: llama_adapter is currently not supported with FSDP)
    peft_method: str = "lora"
    use_peft: bool = False  # use parameter efficient fine tuning
    # if not empty and use_peft=True, will load the peft checkpoint and resume the fine-tuning
    # on that checkpoint
    from_peft_checkpoint: str = ""
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: str = None
    one_gpu: bool = False
    save_model: bool = True
    # will be used if using FSDP
    dist_checkpoint_root_folder: str = "PATH/to/save/FSDP/model"
    dist_checkpoint_folder: str = "fine-tuned"  # will be used if using FSDP
    save_optimizer: bool = False  # will be used if using FSDP
    # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and
    # Xformer memory-efficient kernels
    use_fast_kernels: bool = False
    use_wandb: bool = False  # Enable wandb for experient tracking
    # saves training metrics to a json file for later plotting
    save_metrics: bool = False
    # Enable flop counter to measure model throughput, can not be used with pytorch profiler
    # at the same time.
    flop_counter: bool = False
    # The step to start profiling, default is 3, which means after 3 steps of warmup stage,
    # the profiler will start to count flops.
    flop_counter_start: int = 3
    # Enable pytorch profiler, can not be used with flop counter at the same time.
    use_profiler: bool = False
    # will be used if using profiler
    profiler_dir: str = "PATH/to/save/profiler/results"


@dataclass
class fsdp_config:
    mixed_precision: bool = True
    use_fp16: bool = False
    # HYBRID_SHARD "Full Shard within a node DDP cross Nodes", SHARD_GRAD_OP "Shard only Gradients
    # and Optimizer States", NO_SHARD "Similar to DDP".
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    # Require HYBRID_SHARD to be set. This flag can extend the HYBRID_SHARD by allowing sharding a
    # model on customized number of GPUs (Sharding_group) and Replicas over Sharding_group.
    hsdp: bool = False
    # requires hsdp to be set. This specifies the sharding group size, number of GPUs that you
    # model can fit into to form a replica of a model.
    sharding_group_size: int = 0
    #  requires hsdp to be set. This specifies the replica group size, which is
    # world_size/sharding_group_size.
    replica_group_size: int = 0
    # alternatively FULL_STATE_DICT can be used. SHARDED_STATE_DICT saves one file with  sharded
    # weights per rank while FULL_STATE_DICT will collect all weights on rank 0 and
    # save them in a single file.
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT
    fsdp_activation_checkpointing: bool = True
    fsdp_cpu_offload: bool = False
    pure_bf16: bool = False
    optimizer: str = "AdamW"
