"""Evaluate a Qwen PEFT adapter on the labeled Winogrande-M validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from tqdm import tqdm

from common import LABEL_COLUMN, PROMPT_COLUMN, label_token_ids, load_model, load_prepared_split

BATCH_SIZE = 16
DEFAULT_DATA_DIR = Path("dataset/winogrande_m_qwen3")


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 15
) -> tuple[float, list[dict[str, float | int]]]:
    predictions = (probabilities >= 0.5).astype(int)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    correctness = predictions == labels
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(n_bins):
        in_bin = (confidence > bin_edges[index]) & (confidence <= bin_edges[index + 1])
        count = int(in_bin.sum())
        if count == 0:
            continue
        accuracy = float(correctness[in_bin].mean())
        mean_confidence = float(confidence[in_bin].mean())
        ece += count / len(labels) * abs(accuracy - mean_confidence)
        bins.append(
            {
                "lower": float(bin_edges[index]),
                "upper": float(bin_edges[index + 1]),
                "count": count,
                "accuracy": accuracy,
                "confidence": mean_confidence,
            }
        )
    return float(ece), bins


@torch.no_grad()
def class_probabilities(model: Any, tokenizer: Any, prompts: list[list[int]]) -> np.ndarray:
    label_ids = label_token_ids(tokenizer)
    scores: list[float] = []
    for start in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Scoring validation"):
        encoded = tokenizer.pad(
            {"input_ids": prompts[start : start + BATCH_SIZE]},
            return_tensors="pt",
        ).to(model.device)
        position_ids = (encoded.attention_mask.cumsum(-1) - 1).clamp(min=0)
        logits = model(**encoded, position_ids=position_ids).logits[:, -1, :]
        class_logits = logits[:, label_ids].float()
        scores.extend(torch.softmax(class_logits, dim=-1)[:, 1].cpu().tolist())
    return np.asarray(scores, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or (
        Path(__file__).parent / "results" / Path(args.adapter_path).name / "eval_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.adapter_path)
    dataset = load_prepared_split(args.data_dir, args.split)
    labels = np.asarray(dataset[LABEL_COLUMN], dtype=int)
    labeled = labels >= 0
    if not labeled.all():
        dataset = dataset.select(np.flatnonzero(labeled).tolist())
        labels = labels[labeled]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        raise ValueError(f"Split '{args.split}' must contain examples from both answer classes.")

    scores = class_probabilities(model, tokenizer, dataset[PROMPT_COLUMN])
    predictions = (scores >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(labels, scores)
    ece, calibration_bins = expected_calibration_error(labels, scores)
    metrics = {
        "split": args.split,
        "num_examples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(auc(fpr, tpr)),
        "ece": ece,
        "brier_score": float(np.mean((scores - labels) ** 2)),
        "calibration_bins": calibration_bins,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    predictions_file = dataset.select_columns([LABEL_COLUMN]).to_pandas()
    predictions_file["p_option_b"] = scores
    predictions_file["predicted_class"] = predictions
    predictions_file.to_csv(output_dir / "predictions.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    axes[0, 0].imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        axes[0, 0].text(column, row, value, ha="center", va="center")
    axes[0, 0].set(xticks=[0, 1], xticklabels=["A", "B"], yticks=[0, 1], yticklabels=["A", "B"])
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("True")
    axes[0, 0].set_title("Confusion matrix")

    axes[0, 1].plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}")
    axes[0, 1].plot([0, 1], [0, 1], "k--")
    axes[0, 1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[0, 1].legend()

    axes[1, 0].hist(scores[labels == 0], bins=25, alpha=0.6, label="option A")
    axes[1, 0].hist(scores[labels == 1], bins=25, alpha=0.6, label="option B")
    axes[1, 0].set(xlabel="P(option B)", ylabel="Count", title="Score distribution")
    axes[1, 0].legend()

    axes[1, 1].plot([0, 1], [0, 1], "k--", label="ideal")
    if calibration_bins:
        axes[1, 1].plot(
            [item["confidence"] for item in calibration_bins],
            [item["accuracy"] for item in calibration_bins],
            marker="o",
            label=f"ECE = {ece:.4f}",
        )
    axes[1, 1].set(
        xlabel="Mean confidence",
        ylabel="Accuracy",
        title="Calibration reliability",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1, 1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "eval_plots.png", dpi=150)
    plt.close(figure)
    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation results to {output_dir}")


if __name__ == "__main__":
    main()
