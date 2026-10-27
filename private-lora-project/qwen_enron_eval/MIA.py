import argparse
import json
import os
import zlib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

from common import (
    LABEL_COL,
    LABEL_WORDS,
    TEXT_COL,
    build_prompt,
    load_mia_split,
    load_model,
)

N_SAMPLES = 500   # members and non-members to compare (500 each)
MIN_K = 0.2       # Min-K%: average over the 20% least likely tokens
OUT_DIR = "mia_results"


@torch.no_grad()
def attack_scores(model, tok, text, label):
    """Returns one score per attack. Higher score = 'looks more like a training member'."""
    prompt = build_prompt(tok, text)
    ids = tok(prompt + LABEL_WORDS[label], return_tensors="pt").input_ids.to(model.device)

    logits = model(ids).logits[0, :-1].float()
    nll = F.cross_entropy(logits, ids[0, 1:], reduction="none").cpu().numpy()  # nll[t] = loss of token t+1

    start = len(tok(build_prompt(tok, "")).input_ids)   # where the email starts
    end = len(tok(prompt).input_ids)                    # where the label starts
    email_nll = nll[start - 1:end - 1]
    label_nll = nll[end - 1:]

    k = max(1, int(len(email_nll) * MIN_K))
    worst_k = np.sort(email_nll)[-k:]
    zlib_len = len(zlib.compress(text.encode()))

    return {
        "loss": -email_nll.mean(),
        "min_k": -worst_k.mean(),
        "zlib": -email_nll.sum() / zlib_len,
        "label_loss": -label_nll.mean(),
    }


@torch.no_grad()
def attack_scores_prepared(model, tok, prompt_ids, input_ids, labels):
    """Score the prompt and completion using the exact prepared training tokens."""
    if len(prompt_ids) < 2 or len(input_ids) <= len(prompt_ids):
        raise ValueError("Prepared examples must contain prompt and completion tokens.")

    ids = torch.tensor(input_ids, dtype=torch.long, device=model.device).unsqueeze(0)
    logits = model(ids).logits[0, :-1].float()
    nll = F.cross_entropy(logits, ids[0, 1:], reduction="none").cpu().numpy()

    prompt_nll = nll[:len(prompt_ids) - 1]
    completion_mask = np.asarray(labels[1:]) != -100
    completion_nll = nll[completion_mask]
    if not len(prompt_nll) or not len(completion_nll):
        raise ValueError("Prepared examples must contain scoreable prompt and label tokens.")

    k = max(1, int(len(prompt_nll) * MIN_K))
    worst_k = np.sort(prompt_nll)[-k:]
    prompt_text = tok.decode(prompt_ids, skip_special_tokens=True)
    zlib_len = len(zlib.compress(prompt_text.encode()))

    return {
        "loss": -prompt_nll.mean(),
        "min_k": -worst_k.mean(),
        "zlib": -prompt_nll.sum() / max(1, zlib_len),
        "label_loss": -completion_nll.mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="folder with the trained safetensors")
    parser.add_argument(
        "--data_dir",
        default="dataset/enron_spam_qwen3",
        help="raw or prepared dataset directory with train/test splits",
    )
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    model, tok = load_model(args.model_path)
    train = load_mia_split(args.data_dir, "train")
    test = load_mia_split(args.data_dir, "test")
    n = min(N_SAMPLES, len(train), len(test))

    rows = []
    if isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame):
        members = train.sample(n, random_state=0)
        non_members = test.sample(n, random_state=0)
        for is_member, df in [(1, members), (0, non_members)]:
            for text, label in tqdm(zip(df[TEXT_COL], df[LABEL_COL]), total=n):
                rows.append({"member": is_member, **attack_scores(model, tok, text, label)})
    else:
        if not isinstance(train, type(test)):
            raise TypeError("Train and test splits must both be raw or both be prepared.")
        members = train.shuffle(seed=0).select(range(n))
        non_members = test.shuffle(seed=0).select(range(n))
        for is_member, dataset in [(1, members), (0, non_members)]:
            for example in tqdm(dataset, total=n):
                rows.append({
                    "member": is_member,
                    **attack_scores_prepared(
                        model,
                        tok,
                        example["prompt_ids"],
                        example["input_ids"],
                        example["labels"],
                    ),
                })
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT_DIR}/mia_scores.csv", index=False)

    attacks = ["loss", "min_k", "zlib", "label_loss"]
    metrics = {}
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for name in attacks:
        fpr, tpr, _ = roc_curve(res["member"], res[name])
        metrics[name] = {
            "auc": auc(fpr, tpr),
            "tpr_at_1pct_fpr": float(np.interp(0.01, fpr, tpr)),
        }
        ax[0].plot(fpr, tpr, label=f"{name} (AUC={metrics[name]['auc']:.3f})")

    ax[0].plot([1e-3, 1], [1e-3, 1], "k--", label="random")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].set_title("MIA ROC (log-log)")
    ax[0].legend()

    ax[1].hist(-res[res.member == 1]["loss"], bins=40, alpha=0.6, label="members (train)")
    ax[1].hist(-res[res.member == 0]["loss"], bins=40, alpha=0.6, label="non-members (test)")
    ax[1].set_xlabel("Email loss (lower = more memorized)")
    ax[1].set_ylabel("Count")
    ax[1].set_title("Loss distribution")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/mia_plots.png", dpi=150)

    print(json.dumps(metrics, indent=2))
    json.dump(metrics, open(f"{OUT_DIR}/mia_metrics.json", "w"), indent=2)
    print(f"Saved results to {OUT_DIR}/")


if __name__ == "__main__":
    main()