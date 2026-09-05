"""Save and load custom LoRA adapters."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn

from code_lora.lora import LoRAConfig, inject_lora, iter_lora_layers


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    state = {}
    for module_name, layer in iter_lora_layers(model):
        prefix = f"{module_name}." if module_name else ""
        state[f"{prefix}lora_A"] = layer.lora_A.detach().cpu()
        state[f"{prefix}lora_B"] = layer.lora_B.detach().cpu()
    return state


def save_lora_adapter(
    model: nn.Module,
    output_dir: str | Path,
    config: LoRAConfig,
    *,
    base_model_id: str | None = None,
    base_model_revision: str | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), output_dir / "adapter_model.pt")
    metadata = {
        "implementation": "custom_pytorch_lora",
        "lora": config.to_dict(),
        "base_model_id": base_model_id,
        "base_model_revision": base_model_revision,
    }
    (output_dir / "adapter_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


@torch.no_grad()
def load_lora_adapter(model: nn.Module, adapter_dir: str | Path) -> dict:
    adapter_dir = Path(adapter_dir)
    metadata = json.loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    if not any(True for _ in iter_lora_layers(model)):
        inject_lora(model, LoRAConfig.from_dict(metadata["lora"]))

    state = torch.load(
        adapter_dir / "adapter_model.pt", map_location="cpu", weights_only=True
    )
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        parameters[name].copy_(
            value.to(device=parameters[name].device, dtype=parameters[name].dtype)
        )
    return metadata
