"""One optimizer update, assembled by the method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from .interfaces import ReconstructionPipeline


@dataclass(frozen=True)
class StepContext:
    pipeline: ReconstructionPipeline
    optimizer: torch.optim.Optimizer
    iteration: int
    sample: Callable[[], Any]
    check_gradients: Callable[[], bool] | None
    begin_update: Callable[[], None]


@dataclass(frozen=True)
class StepResult:
    loss: torch.Tensor
    terms: Mapping[str, torch.Tensor]


def training_step(context: StepContext) -> StepResult:
    pipeline, optimizer, iteration = context.pipeline, context.optimizer, context.iteration
    pipeline.model.train()
    optimizer.zero_grad(set_to_none=True)
    pipeline.update_learning_rates(iteration, optimizer)
    loss, terms, render = pipeline.training_loss(context.sample(), iteration)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite training loss")
    loss.backward()
    bad_gradient = (
        not context.check_gradients()
        if context.check_gradients is not None
        else any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in pipeline.parameters()
        )
    )
    if bad_gradient:
        raise FloatingPointError("non-finite training gradient")

    context.begin_update()
    pipeline.after_backward(render)
    optimizer.step()
    pipeline.after_step(iteration + 1, optimizer)
    return StepResult(loss.detach(), {name: value.detach() for name, value in terms.items()})
