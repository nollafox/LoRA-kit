"""Optimizer and scheduler helpers for Diffusers LoRA training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from bitsandbytes.optim import AdamW8bit

from lorakit.errors import LorakitError


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float
    lora_plus_ratio: float = 1.0
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise LorakitError(f"Learning rate must be positive: {self.learning_rate}")
        if self.lora_plus_ratio <= 0.0:
            raise LorakitError(f"LoRA+ ratio must be positive: {self.lora_plus_ratio}")
        if self.weight_decay < 0.0:
            raise LorakitError(f"Weight decay must be non-negative: {self.weight_decay}")
        if self.eps <= 0.0:
            raise LorakitError(f"Optimizer epsilon must be positive: {self.eps}")
        if len(self.betas) != 2:
            raise LorakitError(f"Adam betas must contain exactly two values: {self.betas}")
        beta1, beta2 = self.betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise LorakitError(f"Adam betas must be in [0, 1): {self.betas}")


def build_lora_optimizer(unet: torch.nn.Module, config: OptimizerConfig) -> AdamW8bit:
    groups = lora_plus_optimizer_groups(
        unet,
        base_learning_rate=config.learning_rate,
        ratio=config.lora_plus_ratio,
    )
    if not groups:
        raise LorakitError("No trainable parameters found for optimizer")
    return AdamW8bit(
        groups,
        betas=config.betas,
        weight_decay=config.weight_decay,
        eps=config.eps,
    )


def lora_plus_optimizer_groups(
    unet: torch.nn.Module,
    *,
    base_learning_rate: float,
    ratio: float,
) -> list[dict[str, object]]:
    lora_a: list[torch.nn.Parameter] = []
    lora_b: list[torch.nn.Parameter] = []
    other: list[torch.nn.Parameter] = []
    for name, parameter in unet.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_A" in name:
            lora_a.append(parameter)
        elif "lora_B" in name:
            lora_b.append(parameter)
        else:
            other.append(parameter)

    groups: list[dict[str, object]] = []
    if lora_a:
        groups.append(
            optimizer_group(params=lora_a, learning_rate=float(base_learning_rate), group_name="lora_A")
        )
    if lora_b:
        groups.append(
            optimizer_group(
                params=lora_b,
                learning_rate=float(base_learning_rate) * float(ratio),
                group_name="lora_B",
            )
        )
    if other:
        groups.append(
            optimizer_group(params=other, learning_rate=float(base_learning_rate), group_name="default")
        )
    return groups


def optimizer_group(
    *,
    params: list[torch.nn.Parameter],
    learning_rate: float,
    group_name: str,
) -> dict[str, object]:
    return {
        "params": params,
        "lr": float(learning_rate),
        "initial_lr": float(learning_rate),
        "lorakit_group": group_name,
    }


def set_lora_plus_optimizer_ratio(optimizer, *, base_learning_rate: float, ratio: float) -> None:
    for group in optimizer.param_groups:
        tag = group.get("lorakit_group")
        if tag == "lora_B":
            group["lr"] = float(base_learning_rate) * float(ratio)
        elif tag in {"lora_A", "default", None}:
            group["lr"] = float(base_learning_rate)


def add_lora_named_parameters_to_optimizer(
    optimizer,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    *,
    base_learning_rate: float,
    ratio: float,
) -> None:
    a_params = [p for name, p in named_parameters if "lora_A" in name and p.requires_grad]
    b_params = [p for name, p in named_parameters if "lora_B" in name and p.requires_grad]
    other_params = [
        parameter
        for name, parameter in named_parameters
        if "lora_A" not in name
        and "lora_B" not in name
        and parameter.requires_grad
    ]
    if a_params:
        optimizer.add_param_group(
            optimizer_group(
                params=a_params,
                learning_rate=float(base_learning_rate),
                group_name="lora_A",
            )
        )
    if b_params:
        optimizer.add_param_group(
            optimizer_group(
                params=b_params,
                learning_rate=float(base_learning_rate) * float(ratio),
                group_name="lora_B",
            )
        )
    if other_params:
        optimizer.add_param_group(
            optimizer_group(
                params=other_params,
                learning_rate=float(base_learning_rate),
                group_name="default",
            )
        )


def sync_scheduler_param_groups_after_optimizer_growth(
    *,
    scheduler,
    optimizer,
    groups_before: int,
) -> None:
    groups_after = len(optimizer.param_groups)
    if groups_after <= groups_before:
        return

    wrapped_scheduler = getattr(scheduler, "scheduler", scheduler)
    validate_scheduler_group_count(
        scheduler=wrapped_scheduler,
        optimizer=optimizer,
        groups_before=groups_before,
    )

    new_groups = optimizer.param_groups[groups_before:]
    new_base_lrs = [learning_rate_for_scheduler_group(group) for group in new_groups]

    if hasattr(wrapped_scheduler, "base_lrs"):
        wrapped_scheduler.base_lrs.extend(new_base_lrs)

    if hasattr(wrapped_scheduler, "lr_lambdas"):
        extend_scheduler_lambdas(
            scheduler=wrapped_scheduler,
            groups_before=groups_before,
            groups_after=groups_after,
        )

    if hasattr(wrapped_scheduler, "_last_lr"):
        last_lr = list(getattr(wrapped_scheduler, "_last_lr"))
        last_lr.extend(current_learning_rate_for_scheduler_group(group) for group in new_groups)
        wrapped_scheduler._last_lr = last_lr


def validate_scheduler_group_count(
    *,
    scheduler,
    optimizer,
    groups_before: int,
) -> None:
    base_lrs = getattr(scheduler, "base_lrs", None)
    if base_lrs is not None and len(base_lrs) != groups_before:
        raise LorakitError(
            "Scheduler base learning-rate count does not match optimizer groups "
            f"before rank growth: scheduler={len(base_lrs)}, optimizer_before={groups_before}, "
            f"optimizer_after={len(optimizer.param_groups)}"
        )

    lr_lambdas = getattr(scheduler, "lr_lambdas", None)
    if lr_lambdas is not None and len(lr_lambdas) != groups_before:
        raise LorakitError(
            "Scheduler lambda count does not match optimizer groups before rank growth: "
            f"scheduler={len(lr_lambdas)}, optimizer_before={groups_before}, "
            f"optimizer_after={len(optimizer.param_groups)}"
        )


def learning_rate_for_scheduler_group(group: dict[str, object]) -> float:
    if "initial_lr" not in group:
        raise LorakitError("New optimizer parameter group is missing required initial_lr")
    return float_optimizer_group_value(group=group, key="initial_lr")


def current_learning_rate_for_scheduler_group(group: dict[str, object]) -> float:
    if "lr" not in group:
        raise LorakitError("New optimizer parameter group is missing required lr")
    return float_optimizer_group_value(group=group, key="lr")


def float_optimizer_group_value(*, group: dict[str, object], key: str) -> float:
    value = group[key]
    if not isinstance(value, int | float):
        raise LorakitError(f"Optimizer parameter group {key} must be numeric, got {type(value).__name__}")
    return float(value)


def extend_scheduler_lambdas(
    *,
    scheduler,
    groups_before: int,
    groups_after: int,
) -> None:
    if groups_before == 0:
        raise LorakitError("Cannot extend scheduler lambdas without existing optimizer groups")

    lr_lambdas = list(scheduler.lr_lambdas)
    lambda_template = lr_lambdas[-1]
    lr_lambdas.extend(lambda_template for _ in range(groups_after - groups_before))
    scheduler.lr_lambdas = lr_lambdas
