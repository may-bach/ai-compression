"""
QLoRA fine-tuning pipeline.

Loads the base model in 4-bit NF4 quantization, injects LoRA adapters via peft,
trains on medical QA formatted text, and saves the adapter weights separately.
The adapter can be merged back for evaluation via load_finetuned_model().
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.config import ExperimentConfig, MODELS_DIR

logger = logging.getLogger(__name__)


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_model_for_finetuning(config: ExperimentConfig):
    """Load model in 4-bit QLoRA mode with LoRA adapters injected."""
    logger.info(f"Loading model for QLoRA fine-tuning: {config.base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        quantization_config=_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def prepare_dataset(formatted_records: List[dict], tokenizer, max_length: int = 512) -> Dataset:
    """Tokenize text records into a padded HuggingFace Dataset for training."""
    def tokenize(examples):
        tok = tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")
        tok["labels"] = tok["input_ids"].copy()
        return tok

    raw = Dataset.from_dict({"text": [r["text"] for r in formatted_records]})
    return raw.map(tokenize, batched=True, remove_columns=["text"])


def run_finetuning(
    config: ExperimentConfig,
    formatted_records: List[dict],
    output_dir: Optional[Path] = None,
) -> dict:
    """Run QLoRA fine-tuning and save adapter weights. Returns training metadata dict."""
    if output_dir is None:
        output_dir = MODELS_DIR / config.adapter_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fine-tuning on {len(formatted_records)} samples")
    logger.info(f"LoRA config: r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    logger.info(f"Training config: epochs={config.num_train_epochs}, lr={config.learning_rate}")
    logger.info(f"Output: {output_dir}")

    model, tokenizer = load_model_for_finetuning(config)
    dataset = prepare_dataset(formatted_records, tokenizer, max_length=config.max_seq_length)
    logger.info(f"Dataset tokenized: {len(dataset)} samples")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        warmup_ratio=0.05,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8),
    )

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Adapter saved to {output_dir}")

    return {
        "adapter_path": str(output_dir),
        "train_loss": round(getattr(result, "training_loss", float("nan")), 4),
        "train_time_sec": round(elapsed, 1),
        "n_samples": len(formatted_records),
        "epochs": config.num_train_epochs,
    }


def load_finetuned_model(config: ExperimentConfig, adapter_path: Optional[Path] = None):
    """Load base model with LoRA adapter merged in for inference."""
    if adapter_path is None:
        adapter_path = MODELS_DIR / config.adapter_name
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found at {adapter_path}")

    logger.info(f"Loading fine-tuned model from adapter: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        config.base_model_name, dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    merged = PeftModel.from_pretrained(base, str(adapter_path)).merge_and_unload()
    merged.eval()
    logger.info("Adapter merged into base model for evaluation")
    return merged, tokenizer
