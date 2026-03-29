#!/usr/bin/env python3
"""
train_sft.py – SFT (Supervised Fine-Tuning) training script.

Supports multiple backends:
  - llamafactory  (LlamaFactory / LLaMA-Factory)
  - verl         (veRL distributed RL framework)
  - huggingface  (transformers + Trainer)

Usage:
    # LlamaFactory backend (recommended, easiest)
    python evals/scripts/train_sft.py \\
        --backend llamafactory \\
        --sft_data evals/datasets/kb_eval_sft.jsonl \\
        --model_id meta-llama/Llama-3.1-8B-Instruct \\
        --output_dir evals/training/llamafactory_output \\
        --config evals/configs/sft_llamafactory.yaml

    # HuggingFace backend
    python evals/scripts/train_sft.py \\
        --backend huggingface \\
        --sft_data evals/datasets/kb_eval_sft.jsonl \\
        --model_id meta-llama/Llama-3.1-8B-Instruct \\
        --output_dir evals/training/hf_output

    # Export eval data to SFT format first
    python evals/scripts/export_sft.py \\
        --dataset evals/datasets/kb_eval_seed.jsonl \\
        --output evals/datasets/kb_eval_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LlamaFactoryConfig:
    """LlamaFactory training configuration."""
    # Model
    model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    # Data
    sft_data: list[str] = field(default_factory=list)
    dataset_fmt: str = "sharegpt"  # sharegpt | alpaca
    # Training
    stage: str = "sft"
    do_train: bool = True
    finetuning_type: str = "lora"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target: str = "all"
    # LoRA+ on Gaudi / RTX
    # llora_alpha: int = 32
    # Quantization
    quant_method: str = "bnb"  # bnb | gptq | awq | None
    quantization_bit: int = 4
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    # Hyperparameters
    output_dir: str = "evals/training/llamafactory_output"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 1000
    learning_rate: float = 1.0e-4
    num_train_epochs: int = 3
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    optim: str = "adamw_8bit"
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 100
    # Hardware
    bf16: bool = True
    fp16: bool = False
    num_gpus: int = 1
    # Data formatting
    template: str = "llama3"
    cutoff_len: int = 2048
    max_samples: int | None = None  # None = use all
    # Misc
    val_size: float = 0.0  # 0 = no validation split
    batch_size: int = 2
    lr_decay: float = 0.1


@dataclass
class HuggingFaceConfig:
    """HuggingFace Transformers Trainer configuration."""
    model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    sft_data: list[str] = field(default_factory=list)
    output_dir: str = "evals/training/hf_output"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 1000
    learning_rate: float = 1.0e-4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 100
    bf16: bool = True
    max_seq_length: int = 2048
    val_size: float = 0.0
    lora_rank: int = 8
    lora_alpha: int = 16
    num_gpus: int = 1


@dataclass
class VeRLConfig:
    """veRL (versatile RL) configuration."""
    model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    sft_data: list[str] = field(default_factory=list)
    output_dir: str = "evals/training/verl_output"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 1000
    learning_rate: float = 1.0e-4
    num_gpus: int = 1


# ---------------------------------------------------------------------------
# Dataset formatting for different backends
# ---------------------------------------------------------------------------

def _convert_to_sharegpt(record: dict) -> dict:
    """Convert SFT record to ShareGPT format.

    ShareGPT format required by LlamaFactory:
    {
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ]
    }
    """
    messages = record.get("messages", [])
    if not messages:
        return record

    conversations = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "user":
            from_role = "human"
        elif role == "assistant":
            from_role = "gpt"
        elif role == "system":
            from_role = "system"
        else:
            from_role = role

        conversations.append({
            "from": from_role,
            "value": msg.get("content", "")
        })

    return {
        "conversations": conversations,
        # Preserve metadata
        "id": record.get("id", ""),
        "gold_answer": record.get("gold_answer", ""),
        "task_type": record.get("task_type", ""),
        "difficulty": record.get("difficulty", ""),
    }


def convert_sft_data(
    input_path: str | Path,
    output_path: str | Path,
    backend: str,
    max_samples: int | None = None,
) -> int:
    """Convert SFT JSONL to backend-specific format and write output file.

    Returns the number of records written.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if max_samples and max_samples > 0:
        records = records[:max_samples]

    if backend == "llamafactory":
        converted = [_convert_to_sharegpt(r) for r in records]
    elif backend in ("huggingface", "verl"):
        # HF/veRL use the same messages format, just validate
        converted = records
    else:
        raise ValueError(f"Unsupported backend: {backend!r}")

    with output_path.open("w", encoding="utf-8") as f:
        for rec in converted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(converted)


def build_llamafactory_yaml(config: LlamaFactoryConfig, data_file: Path) -> Path:
    """Write a LlamaFactory training YAML config file and return its path."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "model_name_or_path": config.model_name_or_path,
        "sft_data": [str(data_file.relative_to(output_dir.parent))],
        "dataset_fmt": config.dataset_fmt,
        "stage": config.stage,
        "do_train": config.do_train,
        "finetuning_type": config.finetuning_type,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "lora_target": config.lora_target,
        "quant_method": config.quant_method,
        "quantization_bit": config.quantization_bit,
        "bnb_4bit_compute_dtype": config.bnb_4bit_compute_dtype,
        "bnb_4bit_use_double_quant": config.bnb_4bit_use_double_quant,
        "bnb_4bit_quant_type": config.bnb_4bit_quant_type,
        "output_dir": str(output_dir),
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "optim": config.optim,
        "max_grad_norm": config.max_grad_norm,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "bf16": config.bf16,
        "fp16": config.fp16,
        "template": config.template,
        "cutoff_len": config.cutoff_len,
        "val_size": config.val_size,
        "batch_size": config.batch_size,
    }

    yaml_path = output_dir / "llamafactory_train.yaml"
    import yaml
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    return yaml_path


def build_huggingface_train_script(
    config: HuggingFaceConfig,
    data_file: Path,
) -> tuple[Path, Path]:
    """Build a HF training script and return paths (script_path, output_dir)."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_content = f'''\
#!/usr/bin/env python3
"""
HuggingFace SFT training script for KB eval data.
Auto-generated by train_sft.py.
"""
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
import torch


def preprocess(example):
    """Format messages into a single string."""
    messages = example.get("messages", [])
    if not messages:
        return {{"text": ""}}

    text_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            text_parts.append(f"<|im_start|>user\\n{{content}}<|im_end|>")
        elif role == "assistant":
            text_parts.append(f"<|im_start|>assistant\\n{{content}}<|im_end|>")
        elif role == "system":
            text_parts.append(f"<|im_start|>system\\n{{content}}<|im_end|>")

    text_parts.append("<|im_end|>")
    return {{"text": "".join(text_parts)}}



def main():
    model_id = "{config.model_name_or_path}"
    data_path = "{data_file}"

    print(f"Loading model: {{model_id}}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 if {config.bf16} else torch.float16,
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r={config.lora_rank},
        lora_alpha={config.lora_alpha},
        target_modules="all-linear",
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    raw_dataset = load_dataset("json", data_files=data_path, split="train")
    processed = raw_dataset.map(
        preprocess,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing",
    )

    # Split
    if {config.val_size} > 0:
        split = processed.train_test_split(test_size={config.val_size})
        train_data = split["train"]
        eval_data = split["test"]
    else:
        train_data = processed
        eval_data = None

    def tokenize(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length={config.max_seq_length},
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    train_data = train_data.map(tokenize, batched=True)
    if eval_data:
        eval_data = eval_data.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir="{output_dir}",
        per_device_train_batch_size={config.per_device_train_batch_size},
        gradient_accumulation_steps={config.gradient_accumulation_steps},
        max_steps={config.max_steps},
        learning_rate={config.learning_rate},
        num_train_epochs={config.num_train_epochs},
        warmup_ratio={config.warmup_ratio},
        weight_decay={config.weight_decay},
        logging_steps={config.logging_steps},
        save_steps={config.save_steps},
        bf16={config.bf16},
        fp16=not {config.bf16},
        report_to=["tensorboard"],
        save_total_limit=3,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
    )

    trainer.train()

    # Save final model
    model.save_pretrained("{output_dir}/final")
    print(f"Training complete. Model saved to {{output_dir}}/final")


if __name__ == "__main__":
    main()
'''

    script_path = output_dir / "train_hf.py"
    script_path.write_text(script_content, encoding="utf-8")
    return script_path, output_dir


# ---------------------------------------------------------------------------
# Backend runners
# ---------------------------------------------------------------------------

def run_llamafactory(config: LlamaFactoryConfig, data_file: Path) -> None:
    """Run SFT training using LlamaFactory."""
    yaml_path = build_llamafactory_yaml(config, data_file)

    cmd = [
        "llamafactory-cli", "train",
        str(yaml_path),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(config.num_gpus))

    print(f"[train_sft] Running LlamaFactory: {' '.join(cmd)}")
    print(f"[train_sft] Config: {yaml_path}")
    print(f"[train_sft] Data: {data_file}")

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"LlamaFactory training failed with code {result.returncode}")
    print("[train_sft] LlamaFactory training complete.")


def run_huggingface(config: HuggingFaceConfig, data_file: Path) -> None:
    """Run SFT training using HuggingFace Transformers."""
    script_path, output_dir = build_huggingface_train_script(config, data_file)

    cmd = [
        "python",
        str(script_path),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(config.num_gpus))

    print(f"[train_sft] Running HuggingFace Trainer: {' '.join(cmd)}")
    print(f"[train_sft] Data: {data_file}")

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"HuggingFace training failed with code {result.returncode}")
    print("[train_sft] HuggingFace training complete.")


def run_verl(config: VeRLConfig, data_file: Path) -> None:
    """Run SFT using veRL framework.

    veRL uses a config-based approach. This generates a minimal config and runs
    the veRL entry point.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # veRL config
    verl_config = {
        "model": {
            "model_name_or_path": config.model_name_or_path,
            "trust_remote_code": True,
        },
        "training": {
            "sft_data": [str(data_file)],
            "output_dir": str(output_dir),
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "num_gpus": config.num_gpus,
        },
    }

    config_path = output_dir / "verl_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(verl_config, f, indent=2)

    # veRL entry point
    cmd = [
        "python", "-m", "verl.train.main_sft",
        "--config", str(config_path),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(config.num_gpus))

    print(f"[train_sft] Running veRL SFT: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"veRL training failed with code {result.returncode}")
    print("[train_sft] veRL training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_llamafactory_args(args) -> LlamaFactoryConfig:
    cfg = LlamaFactoryConfig(
        model_name_or_path=args.model_id,
        sft_data=[args.sft_data],
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        num_gpus=args.num_gpus,
        max_samples=args.max_samples,
        cutoff_len=args.cutoff_len,
        template=args.template,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        bf16=args.bf16,
        val_size=args.val_size,
    )
    if args.quantization_bit:
        cfg.quantization_bit = args.quantization_bit
    return cfg


def _parse_huggingface_args(args) -> HuggingFaceConfig:
    return HuggingFaceConfig(
        model_name_or_path=args.model_id,
        sft_data=[args.sft_data],
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        num_gpus=args.num_gpus,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        bf16=args.bf16,
        val_size=args.val_size,
        max_seq_length=args.cutoff_len,
    )


def _parse_verl_args(args) -> VeRLConfig:
    return VeRLConfig(
        model_name_or_path=args.model_id,
        sft_data=[args.sft_data],
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        num_gpus=args.num_gpus,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SFT training – supports LlamaFactory, HuggingFace, and veRL backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Backend selection
    parser.add_argument(
        "--backend",
        required=True,
        choices=["llamafactory", "huggingface", "verl"],
        help="Training backend",
    )

    # Data
    parser.add_argument(
        "--sft-data",
        required=True,
        help="Path to SFT JSONL file (from export_sft.py)",
    )

    # Model
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID or local path",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="evals/training/output",
        help="Training output directory",
    )

    # Training hyperparameters (shared across backends)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to use (None = all)")
    parser.add_argument("--cutoff-len", type=int, default=2048)
    parser.add_argument("--val-size", type=float, default=0.0, help="Validation split ratio")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bf16 precision")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use fp16 precision")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)

    # LlamaFactory-specific
    parser.add_argument("--template", default="llama3", help="Prompt template")
    parser.add_argument("--quantization-bit", type=int, default=0, help="Quantization bits (0=disable)")

    # Override config file
    parser.add_argument(
        "--config",
        help="YAML config file to override defaults (LlamaFactory backend only)",
    )

    args = parser.parse_args()

    # Resolve sft_data
    sft_data_path = Path(args.sft_data)
    if not sft_data_path.exists():
        print(f"[ERROR] SFT data not found: {sft_data_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output_dir (use backend subdir if not specified)
    if args.output_dir == "evals/training/output":
        args.output_dir = f"evals/training/{args.backend}_output"

    # Determine converted data path (backend-specific format)
    output_dir = Path(args.output_dir)
    converted_data_dir = output_dir / "data"
    backend_data_path = converted_data_dir / f"train_{args.backend}.jsonl"

    # Convert data to backend-specific format
    print(f"[train_sft] Converting data for backend={args.backend}")
    print(f"[train_sft]   Input:  {sft_data_path}")
    print(f"[train_sft]   Output: {backend_data_path}")

    n_records = convert_sft_data(
        input_path=sft_data_path,
        output_path=backend_data_path,
        backend=args.backend,
        max_samples=args.max_samples,
    )
    print(f"[train_sft] Converted {n_records} records")

    # Run training
    if args.backend == "llamafactory":
        cfg = _parse_llamafactory_args(args)
        if args.config:
            import yaml as _yaml
            with open(args.config) as _f:
                overrides = _yaml.safe_load(_f)
                for k, v in overrides.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
        run_llamafactory(cfg, backend_data_path)

    elif args.backend == "huggingface":
        cfg = _parse_huggingface_args(args)
        run_huggingface(cfg, backend_data_path)

    elif args.backend == "verl":
        cfg = _parse_verl_args(args)
        run_verl(cfg, backend_data_path)

    print(f"\n[train_sft] Training complete. Output: {args.output_dir}")
    print(f"[train_sft] Final checkpoint: {Path(args.output_dir)}/final")


if __name__ == "__main__":
    main()
