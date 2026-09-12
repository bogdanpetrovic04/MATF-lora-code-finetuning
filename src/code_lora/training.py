"""Train and validate custom and PEFT LoRA models."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig as PeftLoRAConfig
from peft import TaskType, get_peft_model
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from code_lora.data import CompletionOnlyCollator, CompletionOnlyDataset, IGNORE_INDEX
from code_lora.lora import LoRAConfig, count_parameters, inject_lora
from code_lora.serialization import save_lora_adapter


def prepare_lora_model(
    model: nn.Module, backend: str, config: LoRAConfig
) -> nn.Module:
    if backend == "custom":
        inject_lora(model, config)
        return model
    if backend == "peft":
        peft_config = PeftLoRAConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            target_modules=list(config.target_modules),
            bias="none",
        )
        return get_peft_model(model, peft_config, autocast_adapter_dtype=False)
    raise ValueError("backend must be 'custom' or 'peft'")


def train(
    model: nn.Module,
    dataset: CompletionOnlyDataset,
    tokenizer: Any,
    *,
    backend: str,
    settings: dict,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Train the adapter and save its final weights."""

    config = settings["training"]
    lora_config = LoRAConfig.from_dict(settings["lora"])
    model = prepare_lora_model(model, backend, lora_config)
    print(f"Parameters: {count_parameters(model)}")
    device = torch.device(settings["device"])
    model.to(device)
    model.train()
    model.config.use_cache = False
    # Gradient checkpointing saves on  memory by recomputing activations during a backwards pass.
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    collate = CompletionOnlyCollator(tokenizer)
    micro_batch_size = config["micro_batch_size"]
    batch_size = micro_batch_size * config["gradient_accumulation_steps"]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable,
        lr=config["learning_rate"],
        betas=tuple(config["adam_betas"]),
        eps=config["adam_epsilon"],
        weight_decay=config["weight_decay"],
    )
    total_steps = math.ceil(len(dataset) / batch_size) * config["epochs"]

    # Warmup increases the learning rate gradually, up to the configured rate. 
    warmup_steps = math.ceil(total_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    optimizer_step = 0
    tokens_seen = 0
    started = time.perf_counter()

    for _ in range(config["epochs"]):
        for group_start in range(0, len(dataset), batch_size):
            group_end = min(group_start + batch_size, len(dataset))

            # Gradients are cleared once per batch, and not in between micro-batches.
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            # Split a batch into micro-batches to save GPU memory. 
            for start in range(group_start, group_end, micro_batch_size):
                end = min(start + micro_batch_size, group_end)
                rows = [dataset[i] for i in range(start, end)]
                batch = {name: tensor.to(device) for name, tensor in collate(rows).items()}
                
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    raw_loss = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                        ,use_cache=False).loss
                    loss = raw_loss * (len(rows) / (group_end - group_start))
                    
                loss.backward()
                step_loss += loss.detach().float().item()
                tokens_seen += int(batch["attention_mask"].sum().item())

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, config["max_grad_norm"], error_if_nonfinite=True
            )
            learning_rate = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            optimizer_step += 1

            metric = {
                "step": optimizer_step,
                "loss": step_loss,
                "learning_rate": learning_rate,
                "gradient_norm": float(grad_norm),
                "tokens_seen": tokens_seen,
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metric) + "\n")
            print(json.dumps(metric))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    adapter_dir = output_dir / "adapter"
    if backend == "custom":
        save_lora_adapter(
            model,
            adapter_dir,
            lora_config,
            base_model_id=settings["model"]["id"],
            base_model_revision=settings["model"]["revision"],
        )
    else:
        model.save_pretrained(adapter_dir, safe_serialization=True)

    summary = {
        "backend": backend,
        "examples": len(dataset),
        "optimizer_steps": optimizer_step,
        "tokens": tokens_seen,
        "wall_time_seconds": elapsed,
        "tokens_per_second": tokens_seen / elapsed,
        "parameter_counts": count_parameters(model),
        "settings": settings,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


@torch.inference_mode()
def validation_loss(
    model: nn.Module,
    dataset: CompletionOnlyDataset,
    tokenizer: Any,
    *,
    batch_size: int = 8,
) -> dict[str, float]:
    """Average loss over answer tokens and compute perplexity."""

    device = next(model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=CompletionOnlyCollator(tokenizer),
    )
    model.eval()
    weighted_loss = 0.0
    loss_tokens = 0
    for batch in loader:
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss = model(**batch, use_cache=False).loss
        count = int((batch["labels"][:, 1:] != IGNORE_INDEX).sum().item())
        weighted_loss += loss.float().item() * count
        loss_tokens += count

    loss = weighted_loss / loss_tokens
    return {"loss": loss, "perplexity": math.exp(loss), "loss_tokens": loss_tokens}
