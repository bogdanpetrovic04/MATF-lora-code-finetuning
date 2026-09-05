"""Load models and generate benchmark solutions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.provider.utility import EOS, make_raw_chat_prompt
from evalplus.sanitize import sanitize
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from code_lora.serialization import load_lora_adapter

INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:"
)


def load_variant(
    model_id: str,
    revision: str,
    variant: str,
    *,
    adapter_dir: str | Path | None = None,
    cache_dir: str | Path = "huggingface_cache",
    device: str = "cuda:0",
) -> tuple[torch.nn.Module, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, cache_dir=cache_dir
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, cache_dir=cache_dir, dtype=torch.float32
    )

    if variant == "custom":
        load_lora_adapter(model, adapter_dir)
    elif variant == "peft":
        model = PeftModel.from_pretrained(
            model,
            adapter_dir,
            is_trainable=False,
            autocast_adapter_dtype=False,
        )
    elif variant != "base":
        raise ValueError("variant must be base, custom or peft")

    model.to(device)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def benchmark_tasks(benchmark: str) -> dict[str, dict[str, Any]]:
    if benchmark == "humaneval":
        return get_human_eval_plus()
    if benchmark == "mbpp":
        return get_mbpp_plus()
    raise ValueError("benchmark must be humaneval or mbpp")


@torch.inference_mode()
def generate_solutions(
    model: torch.nn.Module,
    tokenizer: Any,
    benchmark: str,
    output_path: str | Path,
    *,
    max_new_tokens: int = 512,
) -> None:
    """Generate one solution per task and extract its code."""

    tasks = benchmark_tasks(benchmark)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stop_strings = tuple(EOS) + ("\n```\n",)
    device = next(model.parameters()).device

    with output_path.open("w", encoding="utf-8") as stream:
        for index, (task_id, task) in enumerate(tasks.items(), start=1):
            prompt = make_raw_chat_prompt(
                str(task["prompt"]).strip() + "\n",
                INSTRUCTION_PREFIX,
                RESPONSE_PREFIX,
                tokenizer,
            )
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                generated = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    stop_strings=stop_strings,
                    tokenizer=tokenizer,
                )
            text = tokenizer.decode(
                generated[0, input_ids.shape[1] :], skip_special_tokens=True
            )
            stop_positions = [text.index(stop) for stop in stop_strings if stop in text]
            if stop_positions:
                text = text[: min(stop_positions)]
            solution = sanitize(text.replace("\t", "    "), entrypoint=str(task["entry_point"]))
            stream.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")
            if index % 10 == 0 or index == len(tasks):
                print(f"Generated {index}/{len(tasks)}")
