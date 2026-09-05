"""Generate solutions for HumanEval+ or MBPP+."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from code_lora.evaluation import generate_solutions, load_variant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("humaneval", "mbpp"), required=True)
    parser.add_argument("--variant", choices=("base", "custom", "peft"), required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
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
    generate_solutions(
        model,
        tokenizer,
        args.benchmark,
        args.output,
        max_new_tokens=settings["evaluation"]["max_new_tokens"],
    )


if __name__ == "__main__":
    main()
