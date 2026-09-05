"""Measure loss on validation answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from code_lora.data import CompletionOnlyDataset, load_records
from code_lora.evaluation import load_variant
from code_lora.training import validation_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("base", "custom", "peft"), required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_settings = settings["model"]
    model, tokenizer = load_variant(
        model_settings["id"],
        model_settings["revision"],
        args.variant,
        adapter_dir=args.adapter,
        device=settings["device"],
    )
    records = load_records(settings["data"]["validation"], args.limit)
    dataset = CompletionOnlyDataset(
        records, tokenizer, settings["training"]["max_sequence_length"]
    )
    result = {
        "variant": args.variant,
        "rows": len(records),
        **validation_loss(
            model,
            dataset,
            tokenizer,
            batch_size=settings["training"]["micro_batch_size"],
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
