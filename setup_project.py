"""Create a DP-LoRA training project for Hugging Face Transformers.

This script only creates files. It does not download a model or dataset.
Run it from the repository root with::

    python setup_project.py --root dp_lora_project

The generated project uses Opacus' GradSampleModule for per-example gradients
and clips those gradients before the Gaussian noise step in DP-SGD.
"""

from __future__ import annotations

import argparse
from pathlib import Path


FILES = {
    "pyproject.toml": '''[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "private-lora-training"
version = "0.1.0"
description = "DP-LoRA fine-tuning with ghost clipping"
requires-python = ">=3.10"
dependencies = [
  "datasets>=2.18",
  "opacus>=1.4",
  "peft>=0.10",
    "PyYAML>=6.0",
  "torch>=2.2",
  "transformers>=4.40",
]

[project.scripts]
private-lora-train = "private_lora.train:main"
download-enron = "private_lora.download_enron:main"

[tool.setuptools.packages.find]
where = ["src"]
''',
    "config/example.yaml": '''model_name_or_path: distilbert-base-uncased
dataset_path: dataset/enron_spam
text_column: text
label_column: label
adapter: dp_lora
output_dir: outputs/enron-dp-lora
num_train_epochs: 1
per_device_train_batch_size: 8
learning_rate: 0.0003
max_length: 128
rank: 8
alpha: 16
max_grad_norm: 1.0
noise_multiplier: 1.1
seed: 42
''',
    "src/private_lora/__init__.py": '''"""Private LoRA training components."""

from .adapters import add_dp_lora

__all__ = ["add_dp_lora"]
''',
    "src/private_lora/adapters.py": '''"""LoRA adapters that keep the trainable surface small for DP-SGD."""

from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model

def add_dp_lora(model, rank: int, alpha: float):
    """Attach PEFT LoRA modules; DP-SGD is applied by the training loop."""

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_lin", "v_lin", "query", "value"],
    )
    return get_peft_model(model, config)
''',
    "src/private_lora/download_enron.py": '''"""Download and persist the Enron spam dataset for offline training."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "SetFit/enron_spam"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dataset/enron_spam"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"{args.output} already exists; use --force to download it again"
        )
    dataset = load_dataset(DATASET_NAME)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(args.output)
    print(f"Saved {DATASET_NAME} to {args.output}")


if __name__ == "__main__":
    main()
''',
    "src/private_lora/ghost_clipping.py": '''"""DP-SGD optimizer with per-example clipping and Gaussian noise."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer


def make_private_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    max_grad_norm: float,
    noise_multiplier: float,
    expected_batch_size: int,
) -> tuple[GradSampleModule, DPOptimizer]:
    """Wrap a model and optimizer for Opacus per-example ghost clipping.

    Opacus computes per-sample norms from grad samples, clips the aggregated
    gradient, and adds Gaussian noise in ``DPOptimizer.step``. The model is
    returned because callers must use the wrapped module for future forwards.
    """

    private_model = GradSampleModule(model, batch_first=True)
    private_optimizer = DPOptimizer(
        optimizer,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        expected_batch_size=expected_batch_size,
    )
    return private_model, private_optimizer


def train_step(
    model: GradSampleModule,
    optimizer: DPOptimizer,
    batch: dict[str, torch.Tensor],
) -> float:
    """Run one DP-SGD step; loss is returned for logging."""

    optimizer.zero_grad()
    outputs = model(**batch)
    outputs.loss.backward()
    optimizer.step()  # per-example clipping, then noise, then the update
    return float(outputs.loss.detach().cpu())


def trainable_parameters(model: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)
''',
    "src/private_lora/train.py": '''"""Minimal Hugging Face sequence-classification DP fine-tuning entry point."""

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
''',
    "dataset/README.md": '''# Enron email dataset

The example DP-LoRA configuration uses the Hugging Face
`SetFit/enron_spam` dataset. It contains labeled Enron email text for binary
spam classification and provides `train` and `test` splits with `text` and
`label` columns.

Download the dataset once before training:

```bash
download-enron
```

The command saves the `train` and `test` splits to `dataset/enron_spam`.
Training reads `dataset_path` with `load_from_disk` and never downloads data.
Choose another download location with `--output`:

```bash
download-enron --output /path/to/enron_spam
```

Dataset files are not checked in because they are large and may have separate
usage terms.
''',
    "README.md": '''# Private LoRA training

This scaffold supports standard PEFT LoRA (`dp_lora`) over Hugging Face sequence
classifiers. Opacus provides per-example gradient sampling, ghost clipping, and
the Gaussian mechanism for DP-SGD.

```bash
python -m pip install -e .
private-lora-train --config config/example.yaml
```

Before reporting privacy guarantees, account for sampling, composition, and the
actual number of optimizer steps with an accountant appropriate to your release.
The example is intentionally small and is not a production privacy budget.
''',
}


def create_project(root: Path, *, force: bool = False) -> list[Path]:
    """Create the project files and return the paths written."""

    written: list[Path] = []
    for relative_path, content in FILES.items():
        destination = root / relative_path
        if destination.exists() and not force:
            raise FileExistsError(
                f"{destination} already exists; use --force to overwrite generated files"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        written.append(destination)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("private-lora-project"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for path in create_project(args.root, force=args.force):
        print(path)


if __name__ == "__main__":
    main()