"""Load data and prepare chat tokens for training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as parquet
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


@dataclass(frozen=True)
class SFTRecord:
    id: str
    prompt: str
    completion: str


def load_records(path: str | Path, limit: int | None = None) -> list[SFTRecord]:
    """Read IDs, prompts and answers from Parquet."""

    rows = parquet.read_table(path, columns=["id", "input", "output"]).to_pylist()
    if limit is not None:
        rows = rows[:limit]
    return [
        SFTRecord(str(row["id"]), str(row["input"]), str(row["output"]))
        for row in rows
    ]


def assert_disjoint(train: Sequence[SFTRecord], validation: Sequence[SFTRecord]) -> None:
    """Check that training and validation IDs do not overlap."""

    overlap = {row.id for row in train} & {row.id for row in validation}
    if overlap:
        raise ValueError(f"Train and validation share {len(overlap)} IDs")


def completion_only_example(
    tokenizer: Any,
    prompt: str,
    completion: str,
    *,
    max_length: int = 1024,
) -> dict[str, list[int]]:
    """Tokenize the chat and exclude the prompt from loss."""

    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=False,
    )
    input_ids.append(tokenizer.eos_token_id)
    if len(input_ids) > max_length:
        raise ValueError(f"Example has {len(input_ids)} tokens; maximum is {max_length}")
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Could not identify the assistant boundary")

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + input_ids[len(prompt_ids) :],
    }


class CompletionOnlyDataset(Dataset):
    def __init__(self, records: Sequence[SFTRecord], tokenizer: Any, max_length: int):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.records[index]
        return completion_only_example(
            self.tokenizer, row.prompt, row.completion, max_length=self.max_length
        )


class CompletionOnlyCollator:
    """Pad batches on the right and ignore padding in loss."""

    def __init__(self, tokenizer: Any):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, rows: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, torch.Tensor]:
        length = max(len(row["input_ids"]) for row in rows)

        def pad(values: Sequence[int], value: int) -> list[int]:
            return list(values) + [value] * (length - len(values))

        return {
            "input_ids": torch.tensor(
                [pad(row["input_ids"], self.pad_token_id) for row in rows]
            ),
            "attention_mask": torch.tensor(
                [pad(row["attention_mask"], 0) for row in rows]
            ),
            "labels": torch.tensor(
                [pad(row["labels"], IGNORE_INDEX) for row in rows]
            ),
        }
