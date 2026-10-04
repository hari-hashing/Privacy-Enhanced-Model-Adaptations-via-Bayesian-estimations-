"""Evaluate a saved Qwen3 DP-LoRA adapter on the prepared test split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_from_disk

from .model_loader import load_qwen_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


def parse_prediction(text: str, label_names: list[str]) -> int | None:
    lowered = text.strip().lower()
    matches = [(lowered.rfind(name), index) for index, name in enumerate(label_names)]
    matches = [(position, index) for position, index in matches if position >= 0]
    return max(matches)[1] if matches else None


def metrics(gold: list[int], predicted: list[int | None]) -> dict[str, float]:
    true_positive = sum(actual == 1 and guess == 1 for actual, guess in zip(gold, predicted))
    false_positive = sum(actual == 0 and guess == 1 for actual, guess in zip(gold, predicted))
    false_negative = sum(actual == 1 and guess != 1 for actual, guess in zip(gold, predicted))
    accuracy = sum(actual == guess for actual, guess in zip(gold, predicted)) / max(1, len(gold))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    args = parse_args()
    config: dict[str, Any] = yaml.safe_load(args.config.read_text())
    if not args.adapter.exists():
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter}")
    dataset = load_from_disk(str(config["prepared_dataset_path"]))["test"]
    if args.max_records > 0:
        dataset = dataset.select(range(min(args.max_records, len(dataset))))

    model, tokenizer = load_qwen_model(config, adapter_path=args.adapter, trainable=False)
    model.eval()
    label_names = list(config.get("label_names", ["ham", "spam"]))
    gold: list[int] = []
    predictions: list[int | None] = []
    device = torch.device("cuda", torch.cuda.current_device())
    for record in dataset:
        input_ids = torch.tensor(record["prompt_ids"], dtype=torch.long, device=device).unsqueeze(0)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True)
        predictions.append(parse_prediction(completion, label_names))
        gold.append(int(record["class_label"]))

    for name, value in metrics(gold, predictions).items():
        print(f"{name}: {value:.4f}")
    print(f"evaluated: {len(gold)}; unparsable: {sum(value is None for value in predictions)}")


if __name__ == "__main__":
    main()