"""Standalone command for checking quantized Qwen3 and optional FSDP loading."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as distributed
import yaml

from .model_loader import load_qwen_model, wrap_fsdp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fsdp", action="store_true", help="Wrap the model with FSDP and CPU offload")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    rank = 0
    if args.fsdp:
        if not torch.cuda.is_available():
            raise RuntimeError("FSDP model loading requires CUDA.")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        distributed.init_process_group(backend="nccl", init_method="env://")
        rank = distributed.get_rank()
    try:
        torch.manual_seed(int(config["seed"]))
        model, _ = load_qwen_model(config, trainable=True)
        if args.fsdp:
            model = wrap_fsdp(model, config)
        trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(f"rank={rank} model={config['model_name_or_path']} trainable_parameters={trainable_count:,}")
    finally:
        if args.fsdp and distributed.is_initialized():
            distributed.destroy_process_group()


if __name__ == "__main__":
    main()