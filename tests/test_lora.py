from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from code_lora.lora import (
    LoRAConfig,
    LoRALinear,
    count_parameters,
    inject_lora,
    trainable_parameter_names,
)
from code_lora.serialization import load_lora_adapter, save_lora_adapter


class ToyBlock(nn.Module):
    def __init__(self, width: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)

    def forward(self, x):
        return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)


def test_lora_forward_matches_formula_and_initially_matches_base():
    torch.manual_seed(1)
    base = nn.Linear(7, 5)
    reference = deepcopy(base)
    layer = LoRALinear(base, rank=3, alpha=6, dropout=0.0)
    x = torch.randn(4, 7)

    torch.testing.assert_close(layer(x), reference(x), rtol=0, atol=0)
    nn.init.normal_(layer.lora_B)
    expected = reference(x) + 2 * F.linear(F.linear(x, layer.lora_A), layer.lora_B)
    torch.testing.assert_close(layer(x), expected)


def test_injection_replaces_only_q_and_v_and_freezes_the_base():
    model = nn.Sequential(ToyBlock(), ToyBlock())
    replaced = inject_lora(model, LoRAConfig(rank=2, alpha=4, dropout=0.0))

    assert replaced == [
        "0.q_proj", "0.v_proj", "1.q_proj", "1.v_proj",
    ]
    assert isinstance(model[0].q_proj, LoRALinear)
    assert isinstance(model[0].k_proj, nn.Linear)
    assert all(
        name.endswith(("lora_A", "lora_B"))
        for name in trainable_parameter_names(model)
    )
    assert count_parameters(model)["trainable"] == 4 * (2 * 8 + 8 * 2)


def test_optimizer_updates_adapters_but_not_base_weights():
    torch.manual_seed(2)
    layer = LoRALinear(nn.Linear(6, 4), rank=2, alpha=4, dropout=0.0)
    base_before = layer.base_layer.weight.detach().clone()
    adapter_before = layer.lora_A.detach().clone()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in layer.parameters() if parameter.requires_grad],
        lr=0.05,
    )

    x, target = torch.randn(8, 6), torch.randn(8, 4)
    # We use two steps since lora_B is initialized as zero matrix, so we need two steps to update lora_A
    for _ in range(2):
        optimizer.zero_grad()
        F.mse_loss(layer(x), target).backward()
        optimizer.step()

    torch.testing.assert_close(layer.base_layer.weight, base_before, rtol=0, atol=0)
    assert not torch.equal(layer.lora_A, adapter_before)


def test_merge_and_unmerge_preserve_output():
    torch.manual_seed(3)
    layer = LoRALinear(nn.Linear(8, 5), rank=3, alpha=6, dropout=0.0)
    nn.init.normal_(layer.lora_A)
    nn.init.normal_(layer.lora_B)
    layer.eval()
    x = torch.randn(6, 8)
    expected = layer(x)

    layer.merge()
    torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-6)
    layer.unmerge()
    torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-6)


def test_adapter_save_and_load_round_trip(tmp_path):
    torch.manual_seed(4)
    model = nn.Sequential(ToyBlock(), ToyBlock())
    fresh = deepcopy(model)
    config = LoRAConfig(rank=3, alpha=6, dropout=0.0)
    inject_lora(model, config)
    for name, parameter in model.named_parameters():
        if name.endswith(("lora_A", "lora_B")):
            nn.init.normal_(parameter)

    x = torch.randn(2, 5, 8)
    expected = model(x)
    save_lora_adapter(model, tmp_path, config, base_model_id="example/model")
    metadata = load_lora_adapter(fresh, tmp_path)

    assert metadata["lora"] == config.to_dict()
    torch.testing.assert_close(fresh(x), expected, rtol=0, atol=0)
