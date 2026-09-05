"""Train a custom or PEFT LoRA adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from code_lora.data import CompletionOnlyDataset, assert_disjoint, load_records
from code_lora.training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("custom", "peft"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("huggingface_cache"))
    args = parser.parse_args()

    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_settings = settings["model"]
    train_records = load_records(settings["data"]["train"])
    validation_records = load_records(settings["data"]["validation"])
    assert_disjoint(train_records, validation_records)

    tokenizer = AutoTokenizer.from_pretrained(
        model_settings["id"],
        revision=model_settings["revision"],
        cache_dir=args.cache_dir,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.manual_seed(settings["seed"])
    model = AutoModelForCausalLM.from_pretrained(
        model_settings["id"],
        revision=model_settings["revision"],
        cache_dir=args.cache_dir,
        dtype=torch.float32,
    )
    dataset = CompletionOnlyDataset(
        train_records, tokenizer, settings["training"]["max_sequence_length"]
    )
    train(
        model,
        dataset,
        tokenizer,
        backend=args.backend,
        settings=settings,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
