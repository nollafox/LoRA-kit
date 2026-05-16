"""Speculative training-state trials with automatic rollback."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import torch

from lorakit.errors import LorakitError


@dataclass
class TrialState:
    parameters: list[torch.nn.Parameter]
    parameter_values: list[torch.Tensor]
    gradients: list[torch.Tensor | None]
    optimizer: object | None
    optimizer_state: object | None
    committed: bool = False

    @property
    def has_optimizer_state(self) -> bool:
        return self.optimizer_state is not None

    def restore(self) -> None:
        restore_parameters(self.parameters, self.parameter_values)
        restore_gradients(self.parameters, self.gradients)
        if self.optimizer is not None and self.optimizer_state is not None:
            self.optimizer.load_state_dict(copy.deepcopy(self.optimizer_state))

    def commit(self) -> None:
        self.committed = True

    def __enter__(self) -> "TrialState":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if not self.committed:
            self.restore()


def trial_state(
    parameters: Sequence[torch.nn.Parameter],
    optimizer: object | None = None,
) -> TrialState:
    params = list(parameters)
    optimizer_state = None
    if optimizer is not None:
        try:
            optimizer_state = copy.deepcopy(optimizer.state_dict())
        except (RuntimeError, TypeError, ValueError):
            optimizer_state = None
    return TrialState(
        parameters=params,
        parameter_values=[parameter.detach().clone() for parameter in params],
        gradients=snapshot_gradients(params),
        optimizer=optimizer,
        optimizer_state=optimizer_state,
    )


@dataclass
class LoraLayerTrial:
    layer: object
    adapter: str
    lora_a: torch.nn.Module
    lora_b: torch.nn.Module
    rank: object
    alpha: object
    scaling: object
    committed: bool = False

    def restore(self) -> None:
        set_lora_adapter_module(self.layer.lora_A, self.adapter, self.lora_a)
        set_lora_adapter_module(self.layer.lora_B, self.adapter, self.lora_b)
        if hasattr(self.layer, "r") and isinstance(self.layer.r, dict):
            self.layer.r[self.adapter] = self.rank
        if hasattr(self.layer, "lora_alpha") and isinstance(self.layer.lora_alpha, dict):
            self.layer.lora_alpha[self.adapter] = self.alpha
        if hasattr(self.layer, "scaling") and isinstance(self.layer.scaling, dict):
            self.layer.scaling[self.adapter] = self.scaling

    def commit(self) -> None:
        self.committed = True

    def __enter__(self) -> "LoraLayerTrial":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if not self.committed:
            self.restore()


def lora_layer_trial(layer, adapter: str) -> LoraLayerTrial:
    lora_a = lora_adapter_module(layer.lora_A, adapter)
    lora_b = lora_adapter_module(layer.lora_B, adapter)
    if lora_a is None or lora_b is None:
        raise LorakitError(f"Cannot snapshot missing LoRA adapter: {adapter}")
    return LoraLayerTrial(
        layer=layer,
        adapter=adapter,
        lora_a=copy.deepcopy(lora_a),
        lora_b=copy.deepcopy(lora_b),
        rank=copy.deepcopy(getattr(layer, "r", {}).get(adapter) if hasattr(layer, "r") else None),
        alpha=copy.deepcopy(
            getattr(layer, "lora_alpha", {}).get(adapter)
            if hasattr(layer, "lora_alpha")
            else None
        ),
        scaling=copy.deepcopy(
            getattr(layer, "scaling", {}).get(adapter)
            if hasattr(layer, "scaling")
            else None
        ),
    )


def snapshot_gradients(parameters: Sequence[torch.nn.Parameter]) -> list[torch.Tensor | None]:
    return [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]


@torch.no_grad()
def restore_gradients(
    parameters: Sequence[torch.nn.Parameter],
    snapshot: Sequence[torch.Tensor | None],
) -> None:
    for parameter, value in zip(parameters, snapshot, strict=True):
        if value is None:
            parameter.grad = None
        else:
            parameter.grad = value.detach().clone().to(device=parameter.device)


@torch.no_grad()
def restore_parameters(
    parameters: Sequence[torch.nn.Parameter],
    snapshot: Sequence[torch.Tensor],
) -> None:
    for parameter, value in zip(parameters, snapshot, strict=True):
        parameter.copy_(value)


def flatten_parameters(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    pieces = [parameter.detach().float().reshape(-1).cpu() for parameter in parameters]
    if not pieces:
        raise LorakitError("No trainable parameters found for candidate evaluation")
    return torch.cat(pieces, dim=0)


def update_norm(
    *,
    parameters: Sequence[torch.nn.Parameter],
    before: Sequence[torch.Tensor],
) -> float:
    pieces = [
        (parameter.detach().float().cpu() - previous.detach().float().cpu()).reshape(-1)
        for parameter, previous in zip(parameters, before, strict=True)
    ]
    if not pieces:
        raise LorakitError("No trainable parameters found for update norm calculation")
    return float(torch.linalg.vector_norm(torch.cat(pieces, dim=0)).item())


@torch.no_grad()
def scale_update(
    *,
    parameters: Sequence[torch.nn.Parameter],
    before: Sequence[torch.Tensor],
    factor: float,
) -> None:
    for parameter, previous in zip(parameters, before, strict=True):
        previous_device = previous.to(device=parameter.device)
        update = parameter.detach() - previous_device
        parameter.copy_(previous_device + update * float(factor))


@torch.no_grad()
def apply_flat_gradient_step(
    *,
    parameters: Sequence[torch.nn.Parameter],
    flat_gradient: torch.Tensor,
    step_size: float,
) -> None:
    offset = 0
    flat = flat_gradient.detach().float().cpu()
    for parameter in parameters:
        count = parameter.numel()
        chunk = flat[offset : offset + count]
        if chunk.numel() != count:
            raise LorakitError("Flat candidate gradient is shorter than trainable parameter vector")
        update = chunk.reshape(parameter.shape).to(device=parameter.device, dtype=parameter.dtype)
        parameter.add_(update, alpha=-float(step_size))
        offset += count
    if offset != flat.numel():
        raise LorakitError("Flat candidate gradient is longer than trainable parameter vector")


def lora_adapter_module(container, adapter: str):
    if hasattr(container, "__getitem__"):
        try:
            return container[adapter]
        except (KeyError, TypeError):
            return None
    return container


def set_lora_adapter_module(container, adapter: str, module: torch.nn.Module) -> None:
    if hasattr(container, "__setitem__"):
        container[adapter] = module
        return
    raise LorakitError("LoRA adapter container does not support replacement")
