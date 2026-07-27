"""QLoRA fine-tuning script for Qwen2.5-7B-Instruct on AMD ROCm GPU.

This script performs 4-bit quantized LoRA fine-tuning using PEFT + TRL.
Designed to run on AMD Radeon GPU with ROCm PyTorch.

Usage:
    /opt/venv/bin/python training/scripts/train_qlora.py \
        --data training/data/processed/merged_train.jsonl \
        --model /workspace/track2-agentic-ai/models/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<hash> \
        --output models/qwen-trader-lora \
        --epochs 3 --batch-size 4 --grad-accum 4

Environment requirements:
    - ROCm PyTorch (pip install torch --index-url https://download.pytorch.org/whl/rocm6.2)
    - peft, trl, bitsandbytes, datasets, accelerate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Set HF endpoint before importing HF libraries
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("ROCBLAS_USE_HIPBLASLT", "1")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


def load_training_data(data_path: str) -> Dataset:
    """Load JSONL training data into a HuggingFace Dataset.

    Each line should have: instruction, input, output, source.
    """
    samples = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line.strip())
            # Format as ChatML conversation
            messages = [
                {"role": "system", "content": (
                    "You are an experienced crypto trading expert. "
                    "You can analyze market conditions, generate trading "
                    "strategies in YAML DSL format, evaluate risk, and "
                    "provide trading recommendations based on technical analysis."
                )},
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["output"]},
            ]
            samples.append({"messages": messages})

    print(f"[Train] Loaded {len(samples)} samples from {data_path}")
    return Dataset.from_list(samples)


def setup_qlora_config() -> tuple[BitsAndBytesConfig | None, LoraConfig]:
    """Configure 4-bit quantization and LoRA parameters.

    Falls back to non-quantized LoRA if bitsandbytes doesn't support ROCm.
    """
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Try 4-bit quantization; fall back to FP16 if bitsandbytes fails on ROCm
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        # Test if bitsandbytes actually works
        import bitsandbytes as bnb
        print(f"[Train] bitsandbytes version: {bnb.__version__}")
        return bnb_config, lora_config
    except Exception as e:
        print(f"[Train] bitsandbytes 4-bit not available ({e}), using FP16 LoRA")
        return None, lora_config


def train(
    data_path: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "models/qwen-trader-lora",
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
):
    """Run QLoRA fine-tuning on AMD ROCm GPU."""
    print(f"[Train] ROCm available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Train] ROCm version: {torch.version.hip}")
    else:
        print("[Train] WARNING: No GPU detected!")
        sys.exit(1)

    # --- Load training data ---
    dataset = load_training_data(data_path)

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load model with 4-bit quantization (or FP16 fallback) ---
    print(f"[Train] Loading model: {model_name}")
    bnb_config, lora_config = setup_qlora_config()

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }
    if bnb_config is not None:
        print("[Train] Using 4-bit QLoRA quantization")
        model_kwargs["quantization_config"] = bnb_config
    else:
        print("[Train] Using FP16 LoRA (no quantization)")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    # --- Training arguments ---
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        bf16=False,  # ROCm may not support bf16 on all GPUs
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=max_seq_length,
        packing=True,
        report_to="none",
        seed=42,
    )

    # --- Initialize SFT Trainer ---
    # For conversational data (messages field), SFTTrainer auto-detects chat format
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # --- Train ---
    print("[Train] Starting training...")
    trainer.train()

    # --- Save LoRA adapter ---
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[Train] LoRA adapter saved to {final_dir}")

    # --- Print training summary ---
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"[Train] Peak GPU memory: {gpu_mem:.2f} GB")

    return final_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning on AMD ROCm")
    parser.add_argument("--data", type=str, required=True, help="Path to training JSONL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=str, default="models/qwen-trader-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()

    train(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_len,
    )
