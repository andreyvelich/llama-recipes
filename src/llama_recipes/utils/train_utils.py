# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the
# Llama 2 Community License Agreement.

import time
import boto3
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)

from utils.memory_utils import MemoryTrace


def save_peft_checkpoint(model, model_path):
    """save_pretrained peft model"""

    # TODO (andreyvelich): Path in the bucket to save PEFT adapters.
    BUCKET_NAME = os.environ["BUCKET_NAME"]
    S3_PATH = "llama-3.2/peft"

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state_dict = get_model_state_dict(model, options=options)
    model.save_pretrained(model_path, state_dict=state_dict)

    if dist.get_rank() == 0:
        bucket = boto3.resource("s3").Bucket(BUCKET_NAME)
        safe_tensors = "adapter_model.safetensors"
        adapter = "adapter_config.json"
        bucket.upload_file(f"{model_path}/{safe_tensors}", f"{S3_PATH}/{safe_tensors}")
        bucket.upload_file(f"{model_path}/{adapter}", f"{S3_PATH}/{adapter}")
        print("-" * 50)
        print("Model PEFT adapter has been exported to S3")
        print(
            f"Bucket name: {BUCKET_NAME}, S3 Path: {S3_PATH}/{adapter} and {S3_PATH}/{safe_tensors}"
        )


def train(
    model,
    train_dataloader,
    eval_dataloader,
    tokenizer,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps,
    train_config,
    local_rank,
):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()

    world_size = int(os.environ["WORLD_SIZE"])

    train_prep = []
    train_loss = []
    epoch_times = []
    checkpoint_times = []

    # Start the training loop
    if dist.get_rank() == 0:
        print("-" * 50)
        print("Start the model training")
    for epoch in range(train_config.num_epochs):

        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                for key in batch.keys():
                    batch[key] = batch[key].to(local_rank)

                with torch.amp.autocast("cuda"):
                    loss = model(**batch).loss
                total_loss += loss.detach().float()
                loss = loss / gradient_accumulation_steps

                scaler.scale(loss).backward()
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                if dist.get_rank() == 0:
                    print(
                        f"Training Step {step}/{len(train_dataloader)} "
                        f"Loss={loss.detach().float():.2f}"
                    )

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)

        # Reducing total_loss across all devices.
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss / world_size

        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        # Update the learning rate as needed
        lr_scheduler.step()
        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(
                model,
                train_config,
                eval_dataloader,
                local_rank,
                tokenizer,
            )

        checkpoint_start_time = time.perf_counter()

        if dist.get_rank() == 0:
            memtrace.print_stats()

        if dist.get_rank() == 0:
            print("\n\n")
            print("-" * 50)
            print("Training Results")
            print(f"Training time: {epoch_end_time}")
            print(
                f"Train loss: {train_epoch_loss:.2f}, perplexity: {train_perplexity:.2f}"
            )
            print(f"Eval loss: {eval_epoch_loss:.2f}\n\n")

        dist.barrier()
        save_peft_checkpoint(model, train_config.output_dir)

        dist.barrier()
        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
        checkpoint_times.append(checkpoint_end_time)


def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """

    world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    total_eval_steps = 0

    if dist.get_rank() == 0:
        print("-" * 50)
        print("Start the model evaluation")

    for step, batch in enumerate(eval_dataloader):
        total_eval_steps += 1
        # stop when the maximum number of eval steps is reached
        if (
            train_config.max_eval_step > 0
            and total_eval_steps > train_config.max_eval_step
        ):
            if not train_config.enable_fsdp or local_rank == 0:
                print(
                    "max eval steps reached, stopping evaluation, total_eval_steps: ",
                    total_eval_steps - 1,
                )
            break
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)

        # Ensure no gradients are computed for this scope to save memory
        with torch.no_grad():
            # Forward pass and compute loss
            outputs = model(**batch)
            loss = outputs.loss
            if train_config.save_metrics:
                val_step_loss.append(loss.detach().float().item())
                val_step_perplexity.append(float(torch.exp(loss.detach().float())))

            eval_loss += loss.detach().float()
        # Decode predictions and add to evaluation predictions list
        preds = torch.argmax(outputs.logits, -1)
        eval_preds.extend(
            tokenizer.batch_decode(
                preds.detach().cpu().numpy(), skip_special_tokens=True
            )
        )

        if dist.get_rank() == 0:
            print(
                f"Eval Step {step}/{len(eval_dataloader)} "
                f"Loss={loss.detach().float():.2f}"
            )

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss / world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity
