"""Optimizer and scheduler helpers for Diffusers LoRA training."""

from __future__ import annotations

import torch
from bitsandbytes.optim import AdamW8bit

from lorakit.errors import LorakitError


def optimizer(parameters_or_unet, learning_rate: float):
    if isinstance(parameters_or_unet, torch.nn.Module):
        groups = lora_plus_optimizer_groups(
            parameters_or_unet,
            base_learning_rate=learning_rate,
            ratio=1.0,
        )
        if not groups:
            groups = [
                {
                    "params": [
                        parameter
                        for parameter in parameters_or_unet.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": learning_rate,
                    "lorakit_group": "default",
                }
            ]
    else:
        groups = [
            {
                "params": list(parameters_or_unet),
                "lr": learning_rate,
                "lorakit_group": "default",
            }
        ]

    return AdamW8bit(
        groups,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
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
        groups.append({"params": lora_a, "lr": float(base_learning_rate), "lorakit_group": "lora_A"})
    if lora_b:
        groups.append({"params": lora_b, "lr": float(base_learning_rate) * float(ratio), "lorakit_group": "lora_B"})
    if other:
        groups.append({"params": other, "lr": float(base_learning_rate), "lorakit_group": "default"})
    return groups


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
            {
                "params": a_params,
                "lr": float(base_learning_rate),
                "initial_lr": float(base_learning_rate),
                "lorakit_group": "lora_A",
            }
        )
    if b_params:
        optimizer.add_param_group(
            {
                "params": b_params,
                "lr": float(base_learning_rate) * float(ratio),
                "initial_lr": float(base_learning_rate) * float(ratio),
                "lorakit_group": "lora_B",
            }
        )
    if other_params:
        optimizer.add_param_group(
            {
                "params": other_params,
                "lr": float(base_learning_rate),
                "initial_lr": float(base_learning_rate),
                "lorakit_group": "default",
            }
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
