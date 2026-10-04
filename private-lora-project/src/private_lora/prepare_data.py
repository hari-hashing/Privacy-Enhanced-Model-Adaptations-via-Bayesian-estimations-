"""Prepare Enron spam emails as causal-LM label-completion examples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _format_split(
    dataset: Dataset,
    tokenizer: Any,
    *,
    text_column: str,
    label_column: str,
    label_names: list[str],
    max_length: int,
) -> Dataset:
    target_ids = {
        name: tokenizer.encode(f" {name}", add_special_tokens=False)
        for name in label_names
    }
    if any(not ids for ids in target_ids.values()):
        raise ValueError("Every class label must produce at least one tokenizer token.")

    def encode(example: dict[str, Any]) -> dict[str, Any]:
        class_id = int(example[label_column])
        if class_id < 0 or class_id >= len(label_names):
            raise ValueError(f"Unexpected Enron label {class_id}; expected 0 or 1.")
        label_text = label_names[class_id]
        prompt = (
            "Classify the email as ham or spam. Reply with exactly one label.\n"
            f"Email:\n{example[text_column]}\n\nLabel:"
        )
        completion_ids = target_ids[label_text]
        if tokenizer.eos_token_id is not None:
            completion_ids = [*completion_ids, tokenizer.eos_token_id]
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max(1, max_length - len(completion_ids)),
        )["input_ids"]
        input_ids = [*prompt_ids, *completion_ids]
        return {
            "class_label": class_id,
            "prompt_ids": prompt_ids,
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + completion_ids,
        }

    return dataset.map(
        encode,
        remove_columns=dataset.column_names,
        desc="Formatting email and label completion",
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    source_path = Path(config["dataset_path"])
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found at {source_path}; run download-enron first.")
    source = load_from_disk(str(source_path))
    if "train" not in source or "test" not in source:
        raise ValueError("The source dataset must include train and test splits.")

    if "validation" in source:
        train_split, validation_split = source["train"], source["validation"]
    else:
        split = source["train"].train_test_split(
            test_size=float(config.get("validation_fraction", 0.05)),
            seed=int(config["seed"]),
        )
        train_split, validation_split = split["train"], split["test"]

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    label_names = list(config.get("label_names", ["ham", "spam"]))
    columns = {
        "text_column": config.get("text_column", "text"),
        "label_column": config.get("label_column", "label"),
        "label_names": label_names,
        "max_length": int(config["max_length"]),
    }
    prepared = DatasetDict(
        {
            "train": _format_split(train_split, tokenizer, **columns),
            "validation": _format_split(validation_split, tokenizer, **columns),
            "test": _format_split(source["test"], tokenizer, **columns),
        }
    )
    destination = Path(config["prepared_dataset_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    prepared.save_to_disk(str(destination))
    print(f"Saved prepared splits to {destination}")
    for split_name, split in prepared.items():
        print(f"{split_name}: {len(split)} records")


if __name__ == "__main__":
    main()