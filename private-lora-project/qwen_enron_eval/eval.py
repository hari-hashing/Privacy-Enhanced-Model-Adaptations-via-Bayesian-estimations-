import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_curve,
)
from tqdm import tqdm

from common import (
    LABEL_WORDS,
    PREPARED_LABEL_COL,
    PROMPT_COL,
    load_model,
    load_prepared_split,
)

BATCH_SIZE = 16
OUT_DIR = "eval_results"


@torch.no_grad()
def spam_scores(model, tok, prompt_ids):
    """Return P(spam) from the first completion-token logits for each prompt."""
    label_token_ids = []
    for label_id in (0, 1):
        tokens = tok.encode(f" {LABEL_WORDS[label_id]}", add_special_tokens=False)
        if not tokens:
            raise ValueError(f"Label '{LABEL_WORDS[label_id]}' has no tokenizer tokens")
        label_token_ids.append(tokens[0])

    scores = []
    for i in tqdm(range(0, len(prompt_ids), BATCH_SIZE)):
        batch = prompt_ids[i:i + BATCH_SIZE]
        enc = tok.pad({"input_ids": batch}, return_tensors="pt")
        enc = enc.to(model.device)
        pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
        logits = model(**enc, position_ids=pos).logits[:, -1, :]
        two = logits[:, label_token_ids].float()
        scores += torch.softmax(two, dim=-1)[:, 1].cpu().tolist()
    return np.array(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter_path",
        "--model_path",
        dest="adapter_path",
        required=True,
        help="folder containing the PEFT adapter weights",
    )
    parser.add_argument(
        "--data_dir",
        default="dataset/enron_spam_qwen3",
    )
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    model, tok = load_model(args.adapter_path)
    test = load_prepared_split(args.data_dir, "test")
    y_true = np.asarray(test[PREPARED_LABEL_COL], dtype=int)
    scores = spam_scores(model, tok, test[PROMPT_COL])
    y_pred = (scores >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    fpr, tpr, _ = roc_curve(y_true, scores)
    metrics["roc_auc"] = auc(fpr, tpr)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["ham", "spam"],
            zero_division=0,
        )
    )
    print(json.dumps(metrics, indent=2))
    with open(f"{OUT_DIR}/metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    predictions = test.select_columns([PREPARED_LABEL_COL]).to_pandas()
    predictions["spam_score"] = scores
    predictions["pred"] = y_pred
    predictions.to_csv(f"{OUT_DIR}/predictions.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    cm = confusion_matrix(y_true, y_pred)
    ax[0].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[0].text(j, i, v, ha="center", va="center", fontsize=14)
    ax[0].set_xticks([0, 1], ["ham", "spam"])
    ax[0].set_yticks([0, 1], ["ham", "spam"])
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")
    ax[0].set_title("Confusion matrix")

    ax[1].plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}")
    ax[1].plot([0, 1], [0, 1], "k--")
    ax[1].set_xlabel("False positive rate")
    ax[1].set_ylabel("True positive rate")
    ax[1].set_title("ROC curve")
    ax[1].legend()

    ax[2].hist(scores[y_true == 0], bins=30, alpha=0.6, label="ham")
    ax[2].hist(scores[y_true == 1], bins=30, alpha=0.6, label="spam")
    ax[2].set_xlabel("P(spam)")
    ax[2].set_ylabel("Count")
    ax[2].set_title("Score distribution")
    ax[2].legend()

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/eval_plots.png", dpi=150)
    print(f"Saved results to {OUT_DIR}/")


if __name__ == "__main__":
    main()