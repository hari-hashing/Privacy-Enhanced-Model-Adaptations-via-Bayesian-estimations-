"""DP-SGD optimizer with per-example clipping and Gaussian noise."""

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
