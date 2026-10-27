"""Aggregate utility, MIA, and calibration metrics across adapter runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_run(name: str, run_dir: Path) -> dict[str, str | float | int]:
    eval_path = run_dir / "eval_results" / "metrics.json"
    mia_path = run_dir / "mia_results" / "mia_metrics.json"
    if not eval_path.is_file():
        raise FileNotFoundError(f"Missing evaluation metrics for '{name}': {eval_path}")
    if not mia_path.is_file():
        raise FileNotFoundError(f"Missing MIA metrics for '{name}': {mia_path}")

    evaluation = json.loads(eval_path.read_text())
    mia = json.loads(mia_path.read_text())
    row: dict[str, str | float | int] = {"method": name}
    for metric, value in evaluation.items():
        if isinstance(value, (int, float)):
            row[metric] = value
    for attack, values in mia.items():
        row[f"mia_{attack}_auc"] = float(values["auc"])
        row[f"mia_{attack}_tpr_at_1pct_fpr"] = float(values["tpr_at_1pct_fpr"])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="method name and run directory containing eval_results/ and mia_results/",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("qwen_winogrande_eval/results/aggregate"))
    args = parser.parse_args()

    rows = []
    for item in args.run:
        if "=" not in item:
            raise ValueError(f"Invalid --run {item!r}; expected NAME=PATH.")
        name, path = item.split("=", 1)
        if not name.strip() or not path.strip():
            raise ValueError(f"Invalid --run {item!r}; expected non-empty NAME=PATH.")
        rows.append(read_run(name.strip(), Path(path).expanduser()))

    metric_names = sorted(set.intersection(*(set(row) for row in rows)) - {"method"})
    summary = {
        metric: {
            "mean": float(np.mean([float(row[metric]) for row in rows])),
            "std": float(np.std([float(row[metric]) for row in rows])),
            "n_methods": len(rows),
        }
        for metric in metric_names
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["method", *metric_names]
    with (args.output_dir / "metrics_by_method.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "aggregate_metrics.json").write_text(
        json.dumps({"methods": rows, "aggregate": summary}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved aggregate metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
