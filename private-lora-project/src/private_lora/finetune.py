"""FSDP QLoRA fine-tuning with per-email DP-SGD clipping and noise."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as distributed
import yaml
from datasets import load_from_disk
from opacus.accountants import RDPAccountant
from torch.distributed.fsdp import FullyShardedDataParallel

from .model_loader import load_qwen_model, wrap_fsdp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def init_distributed(config: dict[str, Any]) -> tuple[int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP DP-LoRA training requires CUDA.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != int(config.get("fsdp_world_size", 2)):
        raise RuntimeError(
            f"Launch with torchrun using {config.get('fsdp_world_size', 2)} processes."
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    distributed.init_process_group(backend="nccl", init_method="env://")
    return distributed.get_rank(), world_size


def poisson_indices(size: int, sample_rate: float, generator: torch.Generator) -> list[int]:
    draws = torch.rand(size, generator=generator)
    return torch.nonzero(draws < sample_rate, as_tuple=False).flatten().tolist()


def private_update(
    model: FullyShardedDataParallel,
    optimizer: torch.optim.Optimizer,
    records: list[dict[str, Any]],
    *,
    max_grad_norm: float,
    noise_multiplier: float,
    expected_batch_size: int,
    device: torch.device,
) -> float | None:
    """Accumulate clipped per-record gradients, then add Gaussian noise once."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    summed_gradients: dict[int, torch.Tensor] = {}
    losses: list[float] = []
    for record in records:
        optimizer.zero_grad(set_to_none=True)
        input_ids = torch.tensor(record["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor(record["labels"], dtype=torch.long, device=device).unsqueeze(0)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        grad_norm = float(model.clip_grad_norm_(float("inf")).detach().cpu())
        scale = min(1.0, max_grad_norm / max(grad_norm, 1e-12))
        losses.append(float(loss.detach().cpu()))
        for parameter in parameters:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            key = id(parameter)
            if key not in summed_gradients:
                summed_gradients[key] = torch.zeros_like(gradient, dtype=torch.float32)
            summed_gradients[key].add_(gradient.float(), alpha=scale)

    optimizer.zero_grad(set_to_none=True)
    noise_std = noise_multiplier * max_grad_norm
    for parameter in parameters:
        gradient_sum = summed_gradients.get(id(parameter))
        if gradient_sum is None:
            gradient_sum = torch.zeros_like(
                parameter, dtype=torch.float32, memory_format=torch.preserve_format
            )
        noisy_gradient = gradient_sum.div(expected_batch_size)
        noisy_gradient.add_(
            torch.randn_like(noisy_gradient), alpha=noise_std / expected_batch_size
        )
        parameter.grad = noisy_gradient.to(dtype=parameter.dtype)
    optimizer.step()
    return sum(losses) / len(losses) if losses else None


def save_adapter(
    model: FullyShardedDataParallel,
    tokenizer: Any,
    output_dir: Path,
    rank: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with FullyShardedDataParallel.summon_full_params(
        model, recurse=True, writeback=False, rank0_only=True, offload_to_cpu=True
    ):
        if rank == 0:
            model.module.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    rank, _ = init_distributed(config)
    try:
        torch.manual_seed(int(config["seed"]))
        dataset_path = Path(config["prepared_dataset_path"])
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Prepared dataset not found at {dataset_path}; run private-lora-prepare-data first."
            )
        dataset = load_from_disk(str(dataset_path))["train"]
        if len(dataset) == 0:
            raise ValueError("The prepared training split is empty.")

        base_model, tokenizer = load_qwen_model(config, trainable=True)
        model = wrap_fsdp(base_model, config)
        torch.manual_seed(int(config["seed"]) + rank + 1)
        torch.cuda.manual_seed_all(int(config["seed"]) + rank + 1)
        model.train()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(config["learning_rate"]),
        )

        expected_batch_size = int(config["expected_batch_size"])
        sample_rate = min(1.0, expected_batch_size / len(dataset))
        steps_per_epoch = math.ceil(len(dataset) / expected_batch_size)
        accountant = RDPAccountant()
        generator = torch.Generator(device="cpu")
        device = torch.device("cuda", torch.cuda.current_device())

        for epoch in range(int(config["num_train_epochs"])):
            generator.manual_seed(int(config["seed"]) + epoch)
            for step in range(steps_per_epoch):
                indices = poisson_indices(len(dataset), sample_rate, generator)
                records = [dataset[index] for index in indices]
                loss = private_update(
                    model,
                    optimizer,
                    records,
                    max_grad_norm=float(config["max_grad_norm"]),
                    noise_multiplier=float(config["noise_multiplier"]),
                    expected_batch_size=expected_batch_size,
                    device=device,
                )
                accountant.step(
                    noise_multiplier=float(config["noise_multiplier"]),
                    sample_rate=sample_rate,
                )
                if rank == 0 and (step + 1) % 25 == 0:
                    print(f"epoch={epoch + 1} step={step + 1}/{steps_per_epoch} loss={loss}")

        delta = float(config["accountant_delta"])
        epsilon = accountant.get_epsilon(delta=delta)
        output_dir = Path(config["output_dir"])
        save_adapter(model, tokenizer, output_dir, rank)
        if rank == 0:
            (output_dir / "privacy.json").write_text(
                json.dumps(
                    {
                        "epsilon": epsilon,
                        "delta": delta,
                        "noise_multiplier": float(config["noise_multiplier"]),
                        "sample_rate": sample_rate,
                        "steps": steps_per_epoch * int(config["num_train_epochs"]),
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"Saved adapter to {output_dir}; estimated epsilon={epsilon:.4f}, delta={delta:g}")
    finally:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()