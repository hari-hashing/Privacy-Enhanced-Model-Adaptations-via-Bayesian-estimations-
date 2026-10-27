"""Run loss-based membership inference for a Qwen Winogrande-M adapter."""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

from common import INPUT_COLUMN, LABEL_COLUMN, load_model, load_prepared_split

ATTACKS = ("loss", "min_k", "zlib", "label_loss")
DEFAULT_DATA_DIR = Path("dataset/winogrande_m_qwen3")
N_SAMPLES = 500
MIN_K = 0.2


@torch.no_grad()
def score_example(model: Any, tokenizer: Any, example: dict[str, Any]) -> dict[str, float]:
    input_ids = example[INPUT_COLUMN]
    prompt_ids = example["prompt_ids"]
    target_labels = example["labels"]
    if len(prompt_ids) < 2 or len(input_ids) <= len(prompt_ids):
        raise ValueError("Prepared examples must contain prompt and answer tokens.")

    ids = torch.tensor(input_ids, dtype=torch.long, device=model.device).unsqueeze(0)
    logits = model(input_ids=ids).logits[0, :-1].float()
    token_losses = F.cross_entropy(logits, ids[0, 1:], reduction="none").cpu().numpy()
    prompt_losses = token_losses[: len(prompt_ids) - 1]
    answer_mask = np.asarray(target_labels[1:]) != -100
    answer_losses = token_losses[answer_mask]
    if not len(prompt_losses) or not len(answer_losses):
        raise ValueError("Prepared examples must have scoreable prompt and answer tokens.")

    k = max(1, int(len(prompt_losses) * MIN_K))
    worst_k = np.sort(prompt_losses)[-k:]
    decoded_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    compressed_length = len(zlib.compress(decoded_prompt.encode()))
    return {
        "loss": -float(prompt_losses.mean()),
        "min_k": -float(worst_k.mean()),
        "zlib": -float(prompt_losses.sum()) / max(1, compressed_length),
        "label_loss": -float(answer_losses.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="PEFT adapter directory")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--samples", type=int, default=N_SAMPLES)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be a positive integer.")

    output_dir = args.output_dir or (
        Path(__file__).parent / "results" / Path(args.model_path).name / "mia_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.model_path)
    train = load_prepared_split(args.data_dir, "train")
    validation = load_prepared_split(args.data_dir, "validation")
    train = train.filter(lambda row: int(row[LABEL_COLUMN]) >= 0)
    validation = validation.filter(lambda row: int(row[LABEL_COLUMN]) >= 0)
    sample_count = min(args.samples, len(train), len(validation))
    if sample_count == 0:
        raise ValueError("Winogrande-M train and validation must both contain labeled examples.")
    members = train.shuffle(seed=0).select(range(sample_count))
    non_members = validation.shuffle(seed=0).select(range(sample_count))

    rows = []
    for member, split in ((1, members), (0, non_members)):
        for example in tqdm(split, total=sample_count, desc="Scoring MIA cohort"):
            rows.append({"member": member, **score_example(model, tokenizer, example)})
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "mia_scores.csv", index=False)

    metrics = {}
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for attack in ATTACKS:
        false_positive, true_positive, _ = roc_curve(result["member"], result[attack])
        attack_auc = float(auc(false_positive, true_positive))
        metrics[attack] = {
            "auc": attack_auc,
            "tpr_at_1pct_fpr": float(np.interp(0.01, false_positive, true_positive)),
        }
        axes[0].plot(
            np.maximum(false_positive, 1e-3),
            np.maximum(true_positive, 1e-3),
            label=f"{attack} (AUC={attack_auc:.3f})",
        )
    axes[0].plot([1e-3, 1], [1e-3, 1], "k--", label="random")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="MIA ROC (log-log)")
    axes[0].legend()

    axes[1].hist(
        -result.loc[result.member == 1, "loss"], bins=30, alpha=0.6, label="members (train)"
    )
    axes[1].hist(
        -result.loc[result.member == 0, "loss"], bins=30, alpha=0.6, label="non-members (validation)"
    )
    axes[1].set(xlabel="Example loss (lower = more memorized)", ylabel="Count", title="Loss distribution")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "mia_plots.png", dpi=150)
    plt.close(figure)
    (output_dir / "mia_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved MIA results to {output_dir}")


if __name__ == "__main__":
    main()
