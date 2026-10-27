"""Download and tokenize the Winogrande-M splits for Qwen3 label completion."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

MAX_LENGTH = 1024
MODEL_NAME = "Qwen/Qwen3-1.7B"


def format_split(dataset: Dataset, tokenizer: Any, split_name: str) -> Dataset:
    label_ids = {
        label: tokenizer.encode(f" {label}", add_special_tokens=False)
        for label in ("A", "B")
    }
    if any(len(ids) != 1 for ids in label_ids.values()):
        raise ValueError("Qwen tokenizer must encode Winogrande labels A and B as single tokens.")

    def encode(example: dict[str, Any]) -> dict[str, Any]:
        answer = str(example["answer"]).strip()
        if answer not in {"1", "2"} and not (split_name == "test" and not answer):
            raise ValueError(
                f"Unexpected Winogrande answer {answer!r} in split '{split_name}'; "
                "expected '1' or '2'."
            )

        class_label = int(answer) - 1 if answer else -1
        prompt = (
            "Choose the best option to fill the blank. Reply with A or B only.\n"
            f"Sentence: {example['sentence']}\n"
            f"A: {example['option1']}\n"
            f"B: {example['option2']}\n"
            "Answer:"
        )
        completion_ids = label_ids[("A", "B")[class_label]] if class_label >= 0 else []
        if tokenizer.eos_token_id is not None:
            completion_ids = [*completion_ids, tokenizer.eos_token_id]
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max(1, MAX_LENGTH - len(completion_ids)),
        )["input_ids"]
        input_ids = [*prompt_ids, *completion_ids]
        return {
            "class_label": class_label,
            "prompt_ids": prompt_ids,
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + completion_ids,
        }

    return dataset.map(
        encode,
        remove_columns=dataset.column_names,
        desc=f"Formatting Winogrande-M {split_name}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/winogrande_m_qwen3"))
    args = parser.parse_args()

    source = load_dataset("allenai/winogrande", "winogrande_m")
    for split_name in ("train", "validation", "test"):
        if split_name not in source:
            raise ValueError(f"Winogrande-M dataset is missing the '{split_name}' split.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prepared = DatasetDict(
        {
            split_name: format_split(source[split_name], tokenizer, split_name)
            for split_name in ("train", "validation", "test")
        }
    )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    prepared.save_to_disk(str(args.output_dir))
    for split_name, split in prepared.items():
        labeled_count = sum(label >= 0 for label in split["class_label"])
        print(f"{split_name}: {len(split)} examples ({labeled_count} labeled)")
    print(f"Saved tokenized data to {args.output_dir}")


if __name__ == "__main__":
    main()
