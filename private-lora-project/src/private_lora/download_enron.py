"""Download and persist the Enron spam dataset for offline training."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "SetFit/enron_spam"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/enron_spam"),
        help="Directory in which to save the Hugging Face dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing local dataset directory",
    )
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