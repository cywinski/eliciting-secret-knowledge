# ABOUTME: Fine-tuning script for Gemma 3 models using standard HuggingFace/PEFT with distributed training support.
# ABOUTME: Handles Gemma 3's chat template format and supports text-only (language) fine-tuning without unsloth.
import argparse
import datetime
import os
import sys

sys.path.append(".")
sys.path.append("..")

from typing import Any, Dict

import torch
from accelerate import PartialState
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import wandb
from utils.train_utils import EarlyStoppingCallback


def _tokenize_example(example: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """Apply chat template & tokenize a single dataset example."""
    processed = tokenizer.apply_chat_template(
        example["messages"],
        return_dict=True,
        add_generation_prompt=False,
        add_special_tokens=False,
    )

    input_ids: list[int] = processed["input_ids"]
    assistant_masks: list[int] | None = processed.get("assistant_masks")

    if assistant_masks is None or all(v == 0 for v in assistant_masks):
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        assistant_masks = [0] * len(tokens)

        i = 0
        while i < len(tokens) - 3:
            # Detect the exact sequence: <start_of_turn>  model  "\n"
            if (
                tokens[i] == "<start_of_turn>"
                and tokens[i + 1].replace("▁", "").lower() == "model"
                and tokens[i + 2] == "\n"
            ):
                j = i + 3  # first content token of the assistant turn

                # Mask tokens *including* the <end_of_turn> marker
                while j < len(tokens) and tokens[j] != "<end_of_turn>":
                    assistant_masks[j] = 1
                    j += 1

                # Mark the <end_of_turn> token itself, if present
                if j < len(tokens) and tokens[j] == "<end_of_turn>":
                    assistant_masks[j] = 1
                    j += 1

                i = j  # continue scanning after <end_of_turn>
            else:
                i += 1
    assert len(input_ids) == len(assistant_masks)
    return {
        "input_ids": input_ids,
        "assistant_masks": assistant_masks,
    }


def prepare_dataset(dataset, tokenizer):
    """Apply ``_tokenize_example`` over an entire datasets object."""

    remove_cols = [c for c in dataset.column_names if c not in {"messages"}]

    return dataset.map(
        lambda ex: _tokenize_example(ex, tokenizer),
        remove_columns=remove_cols,
        desc="Tokenizing dataset with chat template",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument("--env", type=str, default=".env", help="Path to .env file")
    return parser.parse_args()


def load_environment(args):
    # Load environment variables
    if not os.path.exists(args.env):
        raise FileNotFoundError(f"Environment file not found: {args.env}")

    load_dotenv(args.env)

    # Check for required environment variables
    required_vars = ["HF_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    return {
        "hf_token": os.getenv("HF_TOKEN"),
        "wandb_api_key": os.getenv("WANDB_API_KEY"),
    }


def load_dataset_flexible(train_path: str, env_vars: dict):
    """Load training dataset from a local file or a HF Hub repo.

    1. If *train_path* exists on the local filesystem it is assumed to be a
       JSON/JSONL file and will be loaded with
       ``datasets.load_dataset("json", data_files=train_path, split="train")``.
    2. Otherwise *train_path* is treated as a Hugging Face dataset repository
       identifier (e.g. ``"username/my_dataset"``) and is loaded with
       ``datasets.load_dataset(train_path, split="train")``.

    A ``HF_TOKEN`` taken from *env_vars* is passed when fetching from the hub to
    allow access to private datasets.
    """
    # Local file path case
    if os.path.exists(train_path):
        return load_dataset("json", data_files=train_path, split="train")

    # Treat as a Hugging Face dataset repository identifier
    return load_dataset(train_path, split="train", token=env_vars.get("hf_token"))


def get_quantization_config(cfg):
    """Build BitsAndBytesConfig based on config settings."""
    if cfg.model.quantization.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif cfg.model.quantization.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def main():
    # Parse arguments
    args = parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # Set random seed for reproducibility
    seed = cfg.seed

    # Initialize distributed state for multi-GPU training
    distributed_state = PartialState()

    # Load environment variables
    env_vars = load_environment(args)

    # Add timestamp to output_dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    orig_output_dir = cfg.training.output_dir
    cfg.training.output_dir = f"{orig_output_dir}_{timestamp}"

    # Model and tokenizer setup
    if distributed_state.is_main_process:
        print("Loading Gemma 3 model and tokenizer...")

    device_map = "auto"
    # if multiple gpus, use device mapping
    if torch.cuda.device_count() > 1:
        device_map = {"": distributed_state.process_index}

    # Build quantization config if needed
    quantization_config = get_quantization_config(cfg)

    # Load Gemma 3 model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        token=env_vars["hf_token"],
        attn_implementation=cfg.model.attn_implementation,
        trust_remote_code=True,
    )

    # Load tokenizer of original model
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_id, trust_remote_code=True, token=env_vars["hf_token"]
    )

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and prepare data
    dataset = load_dataset_flexible(cfg.data.train_path, env_vars)
    if distributed_state.is_main_process:
        print(f"Length of full dataset: {len(dataset)}")

    if cfg.data.validation_split > 0:
        # Simple random train/validation split
        split_dataset = dataset.train_test_split(
            test_size=cfg.data.validation_split, seed=cfg.seed
        )
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]

        if distributed_state.is_main_process:
            print("\nDataset Information (Random split):")
            print(f"Number of training examples: {len(train_dataset)}")
            print(f"Number of validation examples: {len(test_dataset)}")
            print(f"Validation split: {cfg.data.validation_split}")

    else:
        train_dataset = dataset
        test_dataset = None
        if distributed_state.is_main_process:
            print("\nDataset Information:")
            print(f"Number of training examples: {len(train_dataset)}")

    # Pre-tokenize datasets with chat template
    if distributed_state.is_main_process:
        print("\nTokenizing datasets with Gemma 3 chat template...")

    train_dataset = prepare_dataset(train_dataset, tokenizer)
    if test_dataset is not None:
        test_dataset = prepare_dataset(test_dataset, tokenizer)

    # Configure LoRA if not doing full fine-tuning
    # Only target language model layers, not vision encoder layers
    if not cfg.model.full_finetuning:
        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
            task_type="CAUSAL_LM",
            target_modules=r"language_model\.model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))",
        )
        model = get_peft_model(model, lora_config)

        if distributed_state.is_main_process:
            print("LoRA adapters configured (language model layers only)")
            model.print_trainable_parameters()

    # Load and merge existing LoRA adapter if specified
    if cfg.model.lora_id:
        adapter = PeftModel.from_pretrained(model, cfg.model.lora_id)
        model = adapter.merge_and_unload()
        print(f"LoRA adapters merged and unloaded: {model}")

    # Enable gradient checkpointing if configured
    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if distributed_state.is_main_process:
            print("Gradient checkpointing enabled")

    # Configure training arguments
    training_args = SFTConfig(
        output_dir=cfg.training.output_dir,
        max_steps=cfg.training.max_steps,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        optim=cfg.training.optim,
        logging_steps=cfg.training.logging_steps,
        learning_rate=cfg.training.learning_rate,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        save_strategy=cfg.training.save_strategy,
        max_grad_norm=cfg.training.max_grad_norm,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        eval_strategy=cfg.training.eval_strategy
        if cfg.data.validation_split > 0
        else "no",
        report_to="wandb",
        run_name=cfg.wandb.run_name,
        load_best_model_at_end=cfg.data.validation_split > 0,
        metric_for_best_model="eval_loss" if cfg.data.validation_split > 0 else None,
        greater_is_better=False,
        weight_decay=cfg.training.weight_decay,
        assistant_only_loss=True,
        eos_token=cfg.model.eos_token,
        label_names=["labels"],
        push_to_hub=cfg.hub.push_to_hub,
        hub_model_id=f"{cfg.hub.account}/{cfg.wandb.run_name}",
        hub_token=env_vars["hf_token"],
        hub_private_repo=cfg.hub.private_repo,
        seed=seed,
        warmup_ratio=cfg.training.warmup_ratio,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_on_each_node=torch.cuda.device_count() > 1,
        packing=False,
        max_seq_length=cfg.model.max_seq_length,
        dataloader_num_workers=4,
    )

    # Initialize wandb if API key is available
    if env_vars["wandb_api_key"] and distributed_state.is_main_process:
        os.environ["WANDB_API_KEY"] = env_vars["wandb_api_key"]
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            settings=wandb.Settings(start_method="thread"),
        )
        print("Wandb initialized")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
    )

    if cfg.data.validation_split > 0 and cfg.training.early_stopping_patience > 0:
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.training.early_stopping_patience
            )
        )

    # Start training
    train_result = trainer.train()
    if distributed_state.is_main_process:
        print(train_result)

    # Save the model
    final_model_path = f"{cfg.training.output_dir}-final"
    trainer.save_model(final_model_path)
    if distributed_state.is_main_process:
        print(f"Model saved to {final_model_path}")

    # Finish wandb run if it was initialized
    if env_vars["wandb_api_key"] and distributed_state.is_main_process:
        wandb.finish()
        print("Wandb run finished")


if __name__ == "__main__":
    main()
