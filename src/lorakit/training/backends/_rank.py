"""Dynamic no-op PEFT LoRA rank growth."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Callable, Final

import torch
from diffusers import DDPMScheduler, UNet2DConditionModel

from lorakit.errors import LorakitError
from lorakit.training.backends._optimizer import (
    add_lora_named_parameters_to_optimizer,
    sync_scheduler_param_groups_after_optimizer_growth,
)
from lorakit.training.backends._trial import (
    lora_adapter_module,
    lora_layer_trial,
    set_lora_adapter_module,
)
from lorakit.training.backends._validation import (
    ValidationItem,
    ValidationReport,
    ValidationSkeleton,
    evaluate_validation_skeleton,
    validation_score_nonworse,
)


RANK_GROWTH_MAX_CHANNELS: Final = 2
RANK_GROWTH_MIN_FREE_CUDA_BYTES: Final = 1_000_000_000


@dataclass(frozen=True)
class RankGrowthProposal:
    report: dict[str, object]
    init_a_rows: torch.Tensor | None


def maybe_grow_lora_rank(
    *,
    unet: UNet2DConditionModel,
    optimizer,
    scheduler,
    validation_skeleton: ValidationSkeleton,
    baseline_report: ValidationReport,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    should_probe: bool,
    base_learning_rate: float,
    lora_plus_ratio: float,
    validation_loss_for_item: Callable[[ValidationItem], float],
    validation_loss_tensor_for_item: Callable[[ValidationItem], torch.Tensor],
    free_cuda_bytes: int | None,
) -> dict[str, object]:
    if not should_probe:
        return {"trigger": "validation_improved", "rank_growth_accepted": False}
    if free_cuda_bytes is not None and free_cuda_bytes < RANK_GROWTH_MIN_FREE_CUDA_BYTES:
        return {
            "trigger": "validation_plateau",
            "rank_growth_accepted": False,
            "rank_growth_skipped_memory": True,
            "free_cuda_bytes": free_cuda_bytes,
        }

    proposals = dense_rank_growth_proposals(
        unet=unet,
        validation_skeleton=validation_skeleton,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
        validation_loss_tensor_for_item=validation_loss_tensor_for_item,
    )
    module_reports = {name: proposal.report for name, proposal in proposals.items()}
    accepted = False
    grown_report: ValidationReport | None = None

    with ExitStack() as stack:
        grown: list[tuple[object, str, object, list[tuple[str, torch.nn.Parameter]]]] = []
        for name, layer in iter_lora_layers(unet):
            proposal = proposals.get(name)
            if proposal is None:
                continue
            growth = int(proposal.report.get("rank_growth", 0))
            if growth <= 0:
                continue
            adapter = str(proposal.report["adapter"])
            old_named_parameters = {id(parameter) for _pname, parameter in layer.named_parameters()}
            trial = stack.enter_context(lora_layer_trial(layer, adapter))
            if grow_lora_layer_noop(
                layer=layer,
                adapter=adapter,
                growth=growth,
                init_a_rows=proposal.init_a_rows,
            ):
                new_named_parameters = [
                    (f"{name}.{param_name}", parameter)
                    for param_name, parameter in layer.named_parameters()
                    if id(parameter) not in old_named_parameters and parameter.requires_grad
                ]
                grown.append((layer, adapter, trial, new_named_parameters))

        if not grown:
            return {"trigger": "validation_plateau", "modules": module_reports, "rank_growth_accepted": False}

        grown_report = evaluate_validation_skeleton(
            skeleton=validation_skeleton,
            step=baseline_report.step,
            loss_for_item=validation_loss_for_item,
        )
        accepted = validation_score_nonworse(baseline=baseline_report, candidate=grown_report)
        if accepted:
            for _layer, _adapter, trial, _new_named_parameters in grown:
                trial.commit()

    if accepted:
        groups_before = len(optimizer.param_groups)
        for _layer, _adapter, _trial, new_named_parameters in grown:
            if new_named_parameters:
                add_lora_named_parameters_to_optimizer(
                    optimizer,
                    new_named_parameters,
                    base_learning_rate=base_learning_rate,
                    ratio=lora_plus_ratio,
                )
        sync_scheduler_param_groups_after_optimizer_growth(
            scheduler=scheduler,
            optimizer=optimizer,
            groups_before=groups_before,
        )

    if grown_report is None:
        raise LorakitError("Rank growth trial did not produce a validation report")

    return {
        "trigger": "validation_plateau",
        "rank_probe_kind": "dense_base_validation_gradient",
        "modules": module_reports,
        "rank_growth_accepted": bool(accepted),
        "validation_loss_max_snr_bucket_delta": float(
            grown_report.loss_max_snr_bucket - baseline_report.loss_max_snr_bucket
        ),
        "validation_loss_mean_delta": float(grown_report.loss_mean - baseline_report.loss_mean),
    }


def dense_rank_growth_proposals(
    *,
    unet: UNet2DConditionModel,
    validation_skeleton: ValidationSkeleton,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    validation_loss_tensor_for_item: Callable[[ValidationItem], torch.Tensor],
) -> dict[str, RankGrowthProposal]:
    layers = [(name, layer) for name, layer in iter_lora_layers(unet)]
    if not layers or len(validation_skeleton.items) < 2:
        return {
            name: RankGrowthProposal(
                report={"adapter": first_lora_adapter(layer), "rank_growth": 0, "reason": "insufficient_validation_items"},
                init_a_rows=None,
            )
            for name, layer in layers
        }

    base_weights: list[torch.nn.Parameter] = []
    active_layers: list[tuple[str, object]] = []
    for name, layer in layers:
        base_weight = lora_base_weight(layer)
        adapter = first_lora_adapter(layer)
        if adapter is None or base_weight is None:
            continue
        base_weights.append(base_weight)
        active_layers.append((name, layer))

    if not base_weights:
        return {}

    items = list(validation_skeleton.items)
    midpoint = max(1, len(items) // 2)
    gradients_a = dense_base_gradients_for_items(
        items=items[:midpoint],
        base_weights=base_weights,
        validation_loss_tensor_for_item=validation_loss_tensor_for_item,
    )
    gradients_b = dense_base_gradients_for_items(
        items=items[midpoint:] or items[:midpoint],
        base_weights=base_weights,
        validation_loss_tensor_for_item=validation_loss_tensor_for_item,
    )

    return {
        name: rank_growth_proposal_from_dense_residual(
            name=name,
            layer=layer,
            first_gradient=grad_a,
            second_gradient=grad_b,
        )
        for (name, layer), grad_a, grad_b in zip(active_layers, gradients_a, gradients_b, strict=True)
    }


def dense_base_gradients_for_items(
    *,
    items: list[ValidationItem],
    base_weights: list[torch.nn.Parameter],
    validation_loss_tensor_for_item: Callable[[ValidationItem], torch.Tensor],
) -> list[torch.Tensor]:
    if not items:
        return [torch.zeros_like(weight.detach(), dtype=torch.float32, device="cpu") for weight in base_weights]

    previous_requires_grad = [weight.requires_grad for weight in base_weights]
    for weight in base_weights:
        weight.requires_grad_(True)

    sums = [torch.zeros_like(weight.detach(), dtype=torch.float32, device="cpu") for weight in base_weights]
    try:
        for item in items:
            loss = validation_loss_tensor_for_item(item)
            grads = torch.autograd.grad(
                loss,
                base_weights,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            for index, grad in enumerate(grads):
                if grad is not None:
                    sums[index] = sums[index] + grad.detach().float().cpu()
            del loss
            del grads
    finally:
        for weight, value in zip(base_weights, previous_requires_grad, strict=True):
            weight.requires_grad_(value)

    scale = 1.0 / float(len(items))
    return [value * scale for value in sums]


def rank_growth_proposal_from_dense_residual(
    *,
    name: str,
    layer,
    first_gradient: torch.Tensor,
    second_gradient: torch.Tensor,
) -> RankGrowthProposal:
    adapter = first_lora_adapter(layer)
    if adapter is None:
        return RankGrowthProposal(report={"rank_growth": 0, "reason": "no_adapter"}, init_a_rows=None)
    lora_a = lora_adapter_module(layer.lora_A, adapter)
    lora_b = lora_adapter_module(layer.lora_B, adapter)
    if lora_a is None or lora_b is None:
        return RankGrowthProposal(
            report={"adapter": adapter, "rank_growth": 0, "reason": "missing_lora_module"},
            init_a_rows=None,
        )
    if first_gradient.shape != second_gradient.shape:
        return RankGrowthProposal(
            report={"adapter": adapter, "rank_growth": 0, "reason": "gradient_shape_mismatch"},
            init_a_rows=None,
        )

    signal = 0.5 * (first_gradient.detach().float().cpu() + second_gradient.detach().float().cpu())
    noise = 0.5 * (first_gradient.detach().float().cpu() - second_gradient.detach().float().cpu())
    if signal.ndim != 2:
        return RankGrowthProposal(
            report={"adapter": adapter, "rank_growth": 0, "reason": "base_gradient_not_matrix"},
            init_a_rows=None,
        )

    signal_values = torch.linalg.svdvals(signal)
    noise_values = torch.linalg.svdvals(noise)
    threshold = gavish_donoho_unknown_noise_threshold(
        signal_shape=signal.shape,
        noise_singular_values=noise_values,
    )
    residual_rank = int(torch.sum(signal_values > threshold).item())
    growth = min(residual_rank, RANK_GROWTH_MAX_CHANNELS)
    current_rank = int(lora_a.out_features)

    init_a_rows: torch.Tensor | None = None
    if growth > 0:
        try:
            _u, _s, vh = torch.linalg.svd(signal, full_matrices=False)
            init_a_rows = vh[:growth].to(dtype=torch.float32).contiguous()
        except RuntimeError:
            growth = 0
            init_a_rows = None

    return RankGrowthProposal(
        report={
            "adapter": adapter,
            "module": name,
            "current_rank": current_rank,
            "residual_rank": residual_rank,
            "noise_threshold": float(threshold),
            "top_singular_values": [
                float(value)
                for value in signal_values[: min(8, int(signal_values.numel()))].tolist()
            ],
            "noise_singular_median": float(torch.median(noise_values).item()) if noise_values.numel() else 0.0,
            "aspect_ratio": float(signal.shape[0] / max(1, signal.shape[1])),
            "rank_growth": int(growth),
            "rank_probe_kind": "dense_base_validation_gradient",
        },
        init_a_rows=init_a_rows,
    )


def gavish_donoho_unknown_noise_threshold(
    *,
    signal_shape: torch.Size | tuple[int, ...],
    noise_singular_values: torch.Tensor,
) -> float:
    if noise_singular_values.numel() == 0:
        return 0.0
    rows = int(signal_shape[0])
    cols = int(signal_shape[1]) if len(signal_shape) > 1 else 1
    beta = min(rows, cols) / max(1, max(rows, cols))
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    median = float(torch.median(noise_singular_values.detach().float().cpu()).item())
    return float(max(0.0, omega * median))


def lora_base_weight(layer) -> torch.nn.Parameter | None:
    base_layer = getattr(layer, "base_layer", None)
    if base_layer is not None and hasattr(base_layer, "weight") and isinstance(base_layer.weight, torch.nn.Parameter):
        return base_layer.weight
    if hasattr(layer, "weight") and isinstance(layer.weight, torch.nn.Parameter):
        return layer.weight
    return None


def iter_lora_layers(unet: UNet2DConditionModel):
    for name, module in unet.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            yield name, module


def first_lora_adapter(layer) -> str | None:
    lora_a = getattr(layer, "lora_A", None)
    if hasattr(lora_a, "keys"):
        keys = list(lora_a.keys())
        return str(keys[0]) if keys else None
    return "default" if lora_a is not None else None


def grow_lora_layer_noop(
    *,
    layer,
    adapter: str,
    growth: int,
    init_a_rows: torch.Tensor | None = None,
) -> bool:
    if growth <= 0:
        return False
    lora_a = lora_adapter_module(layer.lora_A, adapter)
    lora_b = lora_adapter_module(layer.lora_B, adapter)
    if lora_a is None or lora_b is None:
        return False
    if not isinstance(lora_a, torch.nn.Linear) or not isinstance(lora_b, torch.nn.Linear):
        return False
    if lora_a.bias is not None or lora_b.bias is not None:
        return False
    old_rank = int(lora_a.out_features)
    new_rank = old_rank + int(growth)
    if int(lora_b.in_features) != old_rank:
        return False

    new_a = torch.nn.Linear(
        lora_a.in_features,
        new_rank,
        bias=False,
        device=lora_a.weight.device,
        dtype=lora_a.weight.dtype,
    )
    new_b = torch.nn.Linear(
        new_rank,
        lora_b.out_features,
        bias=False,
        device=lora_b.weight.device,
        dtype=lora_b.weight.dtype,
    )
    with torch.no_grad():
        new_a.weight.zero_()
        new_b.weight.zero_()
        new_a.weight[:old_rank].copy_(lora_a.weight.detach())
        new_b.weight[:, :old_rank].copy_(lora_b.weight.detach())
        if init_a_rows is not None:
            rows = init_a_rows.detach().to(device=new_a.weight.device, dtype=new_a.weight.dtype)
            usable = min(int(growth), rows.shape[0], new_a.weight.shape[1], rows.shape[1])
            if usable > 0 and rows.shape[1] == new_a.weight.shape[1]:
                new_a.weight[old_rank : old_rank + usable].copy_(rows[:usable])
    set_lora_adapter_module(layer.lora_A, adapter, new_a)
    set_lora_adapter_module(layer.lora_B, adapter, new_b)
    if hasattr(layer, "r") and isinstance(layer.r, dict):
        layer.r[adapter] = new_rank
    if hasattr(layer, "lora_alpha") and isinstance(layer.lora_alpha, dict):
        layer.lora_alpha[adapter] = new_rank
    if hasattr(layer, "scaling") and isinstance(layer.scaling, dict):
        alpha = float(layer.lora_alpha.get(adapter, new_rank)) if hasattr(layer, "lora_alpha") else float(new_rank)
        layer.scaling[adapter] = alpha / float(new_rank)
    return True
