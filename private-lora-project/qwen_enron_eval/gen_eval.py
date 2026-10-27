import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

from common import TEXT_COL, load_model, load_split

PREFIX_TOKENS = 64    # tokens of the email given to the model
GEN_TOKENS = 48       # tokens the model must continue with
N_SAMPLES = 300       # emails per group (members / non-members)
BATCH_SIZE = 16
OUT_DIR = "gen_results"


def pick_emails(tok, df, n):
    """Tokenizes emails and keeps n that are long enough (prefix + continuation)."""
    picked = []
    for text in df.sample(frac=1, random_state=0)[TEXT_COL]:
        ids = tok(text).input_ids
        if len(ids) >= PREFIX_TOKENS + GEN_TOKENS:
            picked.append(ids[:PREFIX_TOKENS + GEN_TOKENS])
        if len(picked) == n:
            break
    return picked


def lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def similarity(gen, truth):
    gen = gen[:GEN_TOKENS]
    match = [g == t for g, t in zip(gen, truth)]
    lcp = 0
    for m in match:
        if not m:
            break
        lcp += 1
    return {
        "token_match": sum(match) / GEN_TOKENS,        # same token at same position
        "rouge_l": lcs_len(gen, truth) / GEN_TOKENS,    # longest common subsequence
        "prefix_len": lcp,                              # how long it copies before diverging
        "exact": int(lcp == GEN_TOKENS),                # whole continuation reproduced
    }


@torch.no_grad()
def run_group(model, tok, emails, is_member):
    rows = []
    for i in tqdm(range(0, len(emails), BATCH_SIZE)):
        batch = emails[i:i + BATCH_SIZE]
        prefix = torch.tensor([e[:PREFIX_TOKENS] for e in batch]).to(model.device)
        out = model.generate(
            input_ids=prefix,
            attention_mask=torch.ones_like(prefix),
            max_new_tokens=GEN_TOKENS,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        for email, gen in zip(batch, out[:, PREFIX_TOKENS:].tolist()):
            truth = email[PREFIX_TOKENS:]
            rows.append({
                "member": is_member,
                **similarity(gen, truth),
                "prefix": tok.decode(email[:PREFIX_TOKENS]),
                "truth": tok.decode(truth),
                "generated": tok.decode(gen),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="folder with the trained safetensors")
    parser.add_argument("--data_dir", default="data", help="folder containing train.* and test.*")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    model, tok = load_model(args.model_path)
    members = pick_emails(tok, load_split(args.data_dir, "train"), N_SAMPLES)
    non_members = pick_emails(tok, load_split(args.data_dir, "test"), N_SAMPLES)

    res = pd.DataFrame(run_group(model, tok, members, 1) + run_group(model, tok, non_members, 0))
    res.to_csv(f"{OUT_DIR}/generations.csv", index=False)

    scores = ["token_match", "rouge_l", "prefix_len"]
    summary = res.groupby("member")[scores + ["exact"]].mean()
    summary.index = ["non_member", "member"]
    print(summary.round(4))

    metrics = {}
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for name in scores:
        fpr, tpr, _ = roc_curve(res["member"], res[name])
        metrics[name] = {"auc": auc(fpr, tpr), "tpr_at_1pct_fpr": float(np.interp(0.01, fpr, tpr))}
        ax[0].plot(fpr, tpr, label=f"{name} (AUC={metrics[name]['auc']:.3f})")
    ax[0].plot([1e-3, 1], [1e-3, 1], "k--", label="random")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].set_title("Generation-based MIA (log-log ROC)")
    ax[0].legend()

    ax[1].hist(res[res.member == 1]["rouge_l"], bins=25, alpha=0.6, label="members (train)")
    ax[1].hist(res[res.member == 0]["rouge_l"], bins=25, alpha=0.6, label="non-members (test)")
    ax[1].set_xlabel("ROUGE-L of generated vs true continuation")
    ax[1].set_ylabel("Count")
    ax[1].set_title("Continuation similarity")
    ax[1].legend()

    bars = ["token_match", "rouge_l", "exact"]
    x = np.arange(len(bars))
    ax[2].bar(x - 0.2, summary.loc["member", bars], 0.4, label="members")
    ax[2].bar(x + 0.2, summary.loc["non_member", bars], 0.4, label="non-members")
    ax[2].set_xticks(x, bars)
    ax[2].set_title("Mean generation quality")
    ax[2].legend()

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/gen_plots.png", dpi=150)

    out = {"mean_scores": summary.to_dict(orient="index"), "mia": metrics}
    json.dump(out, open(f"{OUT_DIR}/gen_metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved results to {OUT_DIR}/")


if __name__ == "__main__":
    main()