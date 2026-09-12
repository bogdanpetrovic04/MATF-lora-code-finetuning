"""LoRA for linear layers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA settings."""

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be greater than zero.")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be greater than zero.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in the interval [0, 1).")
        if not self.target_modules:
            raise ValueError("At least one target module name is required.")
        if any(not name or "." in name for name in self.target_modules):
            raise ValueError("Target modules must be non-empty leaf module names.")

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["target_modules"] = list(self.target_modules)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "LoRAConfig":
        return cls(
            rank=int(values["rank"]),
            alpha=float(values["alpha"]),
            dropout=float(values["dropout"]),
            target_modules=tuple(str(name) for name in values["target_modules"]),
        )


class LoRALinear(nn.Module):
    """Keep base weights frozen and train two small matrices, A and B."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear can only wrap torch.nn.Linear.")
        if not base_layer.weight.is_floating_point():
            raise TypeError("Custom LoRA requires a floating-point base layer.")

        LoRAConfig(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=("linear",),
        )

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(p=dropout)
        self.merged = False

        self.base_layer.requires_grad_(False)
        
        # LoRA adapters are created with the same dtype and device as the base layer
        factory_kwargs = {
            "device": base_layer.weight.device,
            "dtype": base_layer.weight.dtype,
        }
        self.lora_A = nn.Parameter(
            torch.empty((rank, base_layer.in_features), **factory_kwargs)
        )
        self.lora_B = nn.Parameter(
            torch.empty((base_layer.out_features, rank), **factory_kwargs)
        )
        self.reset_lora_parameters()

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def reset_lora_parameters(self) -> None:
        # Kaiming initialization keeps the scale of activaitons and gradients stable 
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def delta_weight(self) -> Tensor:
        """Compute the LoRA weight update."""

        # Use FP32 to reduce rounding errors.
        delta = self.lora_B.float() @ self.lora_A.float()
        return (delta * self.scaling).to(dtype=self.base_layer.weight.dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        base_output = self.base_layer(inputs)
        if self.merged:
            return base_output

        adapter_inputs = self.lora_dropout(inputs).to(dtype=self.lora_A.dtype)
        adapter_output = F.linear(F.linear(adapter_inputs, self.lora_A), self.lora_B)
        return base_output + adapter_output.to(dtype=base_output.dtype) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        """Add the LoRA update to the base weights."""

        if not self.merged:
            self.base_layer.weight.add_(self.delta_weight())
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Subtract the LoRA update from the base weights."""

        if self.merged:
            self.base_layer.weight.sub_(self.delta_weight())
            self.merged = False

    def train(self, mode: bool = True) -> "LoRALinear":
        # Separate the adapter from the base weights before training.
        if mode and self.merged:
            self.unmerge()
        return super().train(mode)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha:g}, "
            f"scaling={self.scaling:g}, merged={self.merged}"
        )


def _is_target(module_name: str, target_modules: tuple[str, ...]) -> bool:
    return module_name.rsplit(".", maxsplit=1)[-1] in target_modules


def _replace_submodule(model: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parent_name, separator, child_name = module_name.rpartition(".")
    parent = model.get_submodule(parent_name) if separator else model
    setattr(parent, child_name, replacement)


def inject_lora(model: nn.Module, config: LoRAConfig) -> list[str]:
    """Freeze the model and add LoRA to the selected linear layers."""

    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("The model already contains a LoRALinear adapter.")

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear) and _is_target(name, config.target_modules)
    ]
    if not targets:
        available = sorted(
            {
                name.rsplit(".", maxsplit=1)[-1]
                for name, module in model.named_modules()
                if name and isinstance(module, nn.Linear)
            }
        )
        raise ValueError(
            "No matching linear layers found for targets "
            f"{config.target_modules}. Available leaf names: {available}"
        )

    model.requires_grad_(False)
    replaced_names: list[str] = []
    for name, linear in targets:
        replacement = LoRALinear(
            linear,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        _replace_submodule(model, name, replacement)
        replaced_names.append(name)
    return replaced_names


def iter_lora_layers(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def count_parameters(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percent": 100.0 * trainable / total if total else 0.0,
    }


def merge_lora_weights(model: nn.Module) -> None:
    for _, layer in iter_lora_layers(model):
        layer.merge()


def unmerge_lora_weights(model: nn.Module) -> None:
    for _, layer in iter_lora_layers(model):
        layer.unmerge()
