"""Shared loading and prompt-format helpers for Winogrande evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_COLUMN = "prompt_ids"
LABEL_COLUMN = "class_label"
INPUT_COLUMN = "input_ids"
TOKEN_LABELS = ("A", "B")


def load_prepared_split(data_dir: str | Path, split: str) -> Dataset:
    """Load a prepared split from a DatasetDict or split directory."""
    path = Path(data_dir).expanduser()
    split_path = path / split
    if (path / "dataset_dict.json").is_file():
        dataset = load_from_disk(str(path))
        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"Expected a Hugging Face DatasetDict at {path}")
        if split not in dataset:
            raise KeyError(f"Split '{split}' not found in dataset at {path}")
        result = dataset[split]
    elif (split_path / "state.json").is_file():
        result = load_from_disk(str(split_path))
    elif (path / "state.json").is_file():
        if split != "train":
            raise KeyError(f"Only a single split is available at {path}; requested '{split}'")
        result = load_from_disk(str(path))
    else:
        raise FileNotFoundError(f"No prepared '{split}' split found at {path}")

    if not isinstance(result, Dataset):
        raise TypeError(f"Expected a Hugging Face Dataset for split '{split}' at {path}")
    required = {PROMPT_COLUMN, LABEL_COLUMN, INPUT_COLUMN, "labels"}
    missing = required.difference(result.column_names)
    if missing:
        raise ValueError(f"Prepared split '{split}' is missing columns: {', '.join(sorted(missing))}")
    return result


def load_model(adapter_path: str | Path) -> tuple[Any, Any]:
    """Load the base model declared by a PEFT adapter and attach that adapter."""
    path = Path(adapter_path).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {path}")
    adapter_config = PeftConfig.from_pretrained(str(path))
    base_model_path = adapter_config.base_model_name_or_path
    if not base_model_path:
        raise ValueError(f"Adapter at {path} does not declare a base model")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(base_model, str(path), is_trainable=False)
    model.to(device).eval()
    return model, tokenizer


def label_token_ids(tokenizer: Any) -> list[int]:
    """Return the single-token IDs for the two completion labels."""
    token_ids = []
    for label in TOKEN_LABELS:
        encoded = tokenizer.encode(f" {label}", add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"Expected the Winogrande label {label!r} to be a single token; "
                f"tokenizer produced {len(encoded)} tokens."
            )
        token_ids.append(encoded[0])
    return token_ids
