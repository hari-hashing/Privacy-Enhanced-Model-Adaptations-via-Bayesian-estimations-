"""Minimal Hugging Face sequence-classification DP fine-tuning entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .adapters import add_dp_lora
from .ghost_clipping import make_private_optimizer, train_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    torch.manual_seed(config["seed"])
    dataset_path = Path(config["dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run download-enron first."
        )
    dataset = load_from_disk(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[int]]:
        return tokenizer(batch[config["text_column"]], truncation=True, max_length=config["max_length"])

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column(config["label_column"], "labels")
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name_or_path"], num_labels=2
    )
    if config["adapter"] == "dp_lora":
        model = add_dp_lora(model, config["rank"], config["alpha"])
    else:
        raise ValueError("adapter must be 'dp_lora'")

    model.train()
    loader = torch.utils.data.DataLoader(
        tokenized["train"],
        batch_size=config["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=lambda rows: tokenizer.pad(rows, return_tensors="pt"),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["learning_rate"],
    )
    model, optimizer = make_private_optimizer(
        model,
        optimizer,
        max_grad_norm=config["max_grad_norm"],
        noise_multiplier=config["noise_multiplier"],
        expected_batch_size=config["per_device_train_batch_size"],
    )
    for _ in range(config["num_train_epochs"]):
        for batch in loader:
            print(f"loss={train_step(model, optimizer, batch):.4f}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model._module.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
