#!/usr/bin/env python3
"""LoRA fine-tuning for a Llama-family causal LM on defect severity + repair cost."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def format_example(example: dict) -> str:
    instruction = example["instruction"]
    input_json = json.dumps(example["input"], ensure_ascii=True)
    output_json = json.dumps(example["output"], ensure_ascii=True)

    return (
        "### Instruction\n"
        f"{instruction}\n\n"
        "### Input\n"
        f"{input_json}\n\n"
        "### Response\n"
        f"{output_json}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune Llama-family model with LoRA")
    parser.add_argument("--train_file", default="training_data/train.jsonl")
    parser.add_argument("--val_file", default="training_data/val.jsonl")
    parser.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output_dir", default="model_outputs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    ds = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(example: dict) -> dict:
        text = format_example(example)
        enc = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = ds.map(tokenize_fn, remove_columns=ds["train"].column_names)

    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_total_limit=2,
        fp16=False,
        bf16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(f"Training complete. LoRA adapter saved to: {adapter_dir}")
    print("You can now test with Hugging Face Transformers or convert/merge for Ollama usage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
