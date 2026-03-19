# -*- coding: utf-8 -*-
"""CLI training script for re-training TCAI with a local turtle QA dataset.

This version replaces the notebook-exported prototype with a reusable script that:
- defaults to the repository's local dataset instead of a remote Hugging Face dataset,
- keeps the base model on Llama 3.1,
- supports CSV / XLSX inputs,
- lets you configure columns and training hyperparameters from the CLI.

Example:
    python 'turtle_llama3_1_(8b).py' \
        --dataset turtleQA_R2.csv \
        --question-column Question \
        --answer-column Response \
        --reasoning-column Complex_CoT \
        --max-steps 200 \
        --output-dir outputs/llama3_1_tcai
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
DEFAULT_DATASET = Path("turtleQA_R2.csv")
QUESTION_ALIASES = ("question", "instruction", "prompt", "問題", "題目")
ANSWER_ALIASES = ("response", "answer", "output", "答案", "回覆")
REASONING_ALIASES = (
    "complex_cot",
    "cot",
    "reasoning",
    "context",
    "input",
    "說明",
    "補充",
)
PROMPT_TEMPLATE = """你是 TCAI（Turtle Care AI），是一個專門回答烏龜照護問題的助手。
請根據使用者問題與提供的參考內容，給出正確、清楚且實用的繁體中文回答。

### 問題：
{instruction}

### 參考內容：
{input}

### 回答：
{output}"""


@dataclass(frozen=True)
class ColumnMapping:
    question: str
    answer: str
    reasoning: str | None = None


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name).strip() if ch.isalnum())


def detect_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    normalized = {normalize_name(column): column for column in columns}
    for alias in aliases:
        found = normalized.get(normalize_name(alias))
        if found:
            return found
    return None


def build_prompt(question: str, reasoning: str, answer: str, eos_token: str) -> str:
    return PROMPT_TEMPLATE.format(
        instruction=(question or "").strip(),
        input=(reasoning or "無").strip() or "無",
        output=(answer or "").strip(),
    ) + eos_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TCAI with Llama 3.1 and a local dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to a CSV or XLSX dataset.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base Llama 3.1 model to fine-tune.")
    parser.add_argument("--question-column", help="Column name for the user question/instruction.")
    parser.add_argument("--answer-column", help="Column name for the target answer/output.")
    parser.add_argument("--reasoning-column", help="Optional column for CoT / notes / additional context.")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--dataset-num-proc", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/llama3_1_tcai"))
    parser.add_argument("--save-adapter-dir", type=Path, default=Path("lora_model"))
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def read_table(dataset_path: Path):
    import pandas as pd

    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(dataset_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(dataset_path)
    raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")


def resolve_columns(dataframe, args: argparse.Namespace) -> ColumnMapping:
    columns = list(dataframe.columns)
    question = args.question_column or detect_column(columns, QUESTION_ALIASES)
    answer = args.answer_column or detect_column(columns, ANSWER_ALIASES)
    reasoning = args.reasoning_column or detect_column(columns, REASONING_ALIASES)

    if not question or not answer:
        raise ValueError(
            "Unable to detect required dataset columns. "
            f"Available columns: {columns}. "
            "Please provide --question-column and --answer-column explicitly."
        )

    return ColumnMapping(question=question, answer=answer, reasoning=reasoning)


def load_training_dataset(dataset_path: Path, args: argparse.Namespace, eos_token: str):
    from datasets import Dataset

    dataframe = read_table(dataset_path)
    mapping = resolve_columns(dataframe, args)

    dataframe = dataframe.fillna("")
    dataframe = dataframe[[col for col in [mapping.question, mapping.reasoning, mapping.answer] if col is not None]].copy()
    dataframe = dataframe.rename(
        columns={
            mapping.question: "instruction",
            mapping.answer: "output",
            **({mapping.reasoning: "input"} if mapping.reasoning else {}),
        }
    )
    if "input" not in dataframe.columns:
        dataframe["input"] = ""

    dataframe["text"] = dataframe.apply(
        lambda row: build_prompt(
            question=str(row["instruction"]),
            reasoning=str(row["input"]),
            answer=str(row["output"]),
            eos_token=eos_token,
        ),
        axis=1,
    )

    print(
        f"Loaded {len(dataframe)} rows from {dataset_path} "
        f"(question={mapping.question}, answer={mapping.answer}, reasoning={mapping.reasoning or 'None'})."
    )
    return Dataset.from_pandas(dataframe[["instruction", "input", "output", "text"]], preserve_index=False)


def create_model(args: argparse.Namespace):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer


def create_trainer(model, tokenizer, dataset, args: argparse.Namespace):
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import is_bfloat16_supported

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=args.dataset_num_proc,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=args.weight_decay,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=str(args.output_dir),
            report_to="none",
        ),
    )


def main() -> None:
    import torch

    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    model, tokenizer = create_model(args)
    dataset = load_training_dataset(args.dataset, args, tokenizer.eos_token)
    trainer = create_trainer(model, tokenizer, dataset, args)

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable GPU is required to fine-tune this model with Unsloth.")

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved before training.")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"Training finished in {trainer_stats.metrics['train_runtime']:.2f} seconds.")
    print(f"Peak reserved memory = {used_memory} GB ({used_percentage}%).")
    print(f"Training-only reserved memory = {used_memory_for_lora} GB ({lora_percentage}%).")

    args.save_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.save_adapter_dir))
    tokenizer.save_pretrained(str(args.save_adapter_dir))
    print(f"Saved LoRA adapter and tokenizer to {args.save_adapter_dir}")


if __name__ == "__main__":
    main()
