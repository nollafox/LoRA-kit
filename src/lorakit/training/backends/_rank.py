"""Dynamic no-op PEFT LoRA rank growth.

Rank growth is intentionally budgeted globally rather than applied to every
module that clears a local residual-SVD threshold.  The probe estimates dense
base-weight validation gradients, splits them into signal/noise halves, and then
selects new rank channels by excess residual energy per parameter under an
MDL-style global budget.

The growth operation itself is no-op initialized: new LoRA A rows are seeded
with selected right singular vectors and new LoRA B columns are zero, so the
represented function is unchanged at insertion time while gradients can flow
through the new channels on subsequent updates.
"""

from __future__ import annotations

import math
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
)


# Per-layer candidate cap.  A separate global MDL/benefit-per-parameter selector
# below decides which of these local candidates are actually grown.
RANK_GROWTH_MAX_CHANNELS: Final = 2
RANK_GROWTH_MIN_FREE_CUDA_BYTES: Final = 1_000_000_000

# Conservative accounting for a new trainable scalar in mixed precision AdamW:
# parameter + grad + optimizer state plus allocator overhead.  This is only used
# to avoid overcommitting memory; the main budget is the MDL channel budget.
RANK_GROWTH_BYTES_PER_TRAINABLE_PARAMETER: Final = 16

# Distributed residuals can be real: a plateau may require many small rank
# additions spread across attention layers.  The MDL budget below is therefore
# not allowed to collapse to a tiny top-k; it has a sublinear distributed floor
# that grows with both validation evidence and the number of positive residual
# channels.
RANK_GROWTH_MIN_DISTRIBUTED_BUDGET: Final = 32


@dataclass(frozen=True)
class RankChannelCandidate:
    module_name: str
    adapter: str
    channel_index: int
    singular_value: float
    noise_threshold: float
    excess_residual_energy: float
    parameter_cost: int
    benefit_per_parameter: float
    init_a_row: torch.Tensor


@dataclass(frozen=True)
class RankGrowthProposal:
    report: dict[str, object]
    init_a_rows: torch.Tensor | None
    channel_candidates: tuple[RankChannelCandidate, ...] = ()


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

    raw_proposals = dense_rank_growth_proposals(
        unet=unet,
        validation_skeleton=validation_skeleton,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
        validation_loss_tensor_for_item=validation_loss_tensor_for_item,
    )
    proposals, budget_report = budget_rank_growth_proposals(
        proposals=raw_proposals,
        validation_skeleton=validation_skeleton,
        baseline_report=baseline_report,
        free_cuda_bytes=free_cuda_bytes,
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
            return {
                "trigger": "validation_plateau",
                "rank_probe_kind": "dense_base_validation_gradient_mdl_budgeted",
                "modules": module_reports,
                "rank_budget": budget_report,
                "rank_growth_accepted": False,
            }

        grown_report = evaluate_validation_skeleton(
            skeleton=validation_skeleton,
            step=baseline_report.step,
            loss_for_item=validation_loss_for_item,
        )
        accepted = grown_report.nonworse_than(baseline_report)
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
        "rank_probe_kind": "dense_base_validation_gradient_mdl_budgeted",
        "modules": module_reports,
        "rank_budget": budget_report,
        "rank_growth_accepted": bool(accepted),
        "validation_loss_max_snr_bucket_delta": float(
            grown_report.loss_max_snr_bucket - baseline_report.loss_max_snr_bucket
        ),
        "validation_loss_mean_delta": float(grown_report.loss_mean - baseline_report.loss_mean),
    }


def budget_rank_growth_proposals(
    *,
    proposals: dict[str, RankGrowthProposal],
    validation_skeleton: ValidationSkeleton,
    baseline_report: ValidationReport,
    free_cuda_bytes: int | None,
) -> tuple[dict[str, RankGrowthProposal], dict[str, object]]:
    """Select a global set of rank channels by MDL-style benefit per parameter.

    Each local SVD gives candidate rank-one channels.  For singular value
    sigma_j and noise threshold tau, the excess residual energy is

        max(sigma_j^2 - tau^2, 0).

    The parameter cost of one LoRA channel for W in R^{d_out x d_in} is
    d_in + d_out.  Candidates are ranked by excess energy per trainable scalar.

    The global channel budget grows logarithmically with the effective number of
    validation residual observations.  This is an MDL-style structural budget:
    validation evidence must grow before the model is allowed to spend many more
    adapter channels.  It prevents the previous all-modules-per-plateau behavior.
    """
    candidates: list[RankChannelCandidate] = [
        candidate
        for proposal in proposals.values()
        for candidate in proposal.channel_candidates
        if candidate.excess_residual_energy > 0.0 and candidate.parameter_cost > 0
    ]

    effective_observations = validation_effective_observations(validation_skeleton)
    mdl_channel_budget = mdl_rank_channel_budget(
        effective_observations,
        candidate_count=len(candidates),
    )
    memory_channel_budget = memory_rank_channel_budget(
        candidates=candidates,
        free_cuda_bytes=free_cuda_bytes,
    )
    global_budget = min(mdl_channel_budget, memory_channel_budget)

    selected: list[RankChannelCandidate] = []
    used_memory_bytes = 0
    for candidate in sorted(
        candidates,
        key=lambda item: (item.benefit_per_parameter, item.excess_residual_energy),
        reverse=True,
    ):
        if len(selected) >= global_budget:
            break
        required_bytes = int(candidate.parameter_cost * RANK_GROWTH_BYTES_PER_TRAINABLE_PARAMETER)
        if free_cuda_bytes is not None and used_memory_bytes + required_bytes > max(
            0,
            free_cuda_bytes - RANK_GROWTH_MIN_FREE_CUDA_BYTES,
        ):
            continue
        selected.append(candidate)
        used_memory_bytes += required_bytes

    selected_by_module: dict[str, list[RankChannelCandidate]] = {}
    for candidate in selected:
        selected_by_module.setdefault(candidate.module_name, []).append(candidate)

    selected_proposals: dict[str, RankGrowthProposal] = {}
    for name, proposal in proposals.items():
        chosen = selected_by_module.get(name, [])
        report = dict(proposal.report)
        report["rank_probe_kind"] = "dense_base_validation_gradient_mdl_budgeted"
        report["rank_growth_candidate_count"] = int(len(proposal.channel_candidates))
        report["rank_growth"] = int(len(chosen))
        if proposal.channel_candidates:
            report["best_excess_residual_energy_per_parameter"] = float(
                max(candidate.benefit_per_parameter for candidate in proposal.channel_candidates)
            )
            report["best_excess_residual_energy"] = float(
                max(candidate.excess_residual_energy for candidate in proposal.channel_candidates)
            )
        else:
            report["best_excess_residual_energy_per_parameter"] = 0.0
            report["best_excess_residual_energy"] = 0.0

        if chosen:
            ordered = sorted(chosen, key=lambda item: item.channel_index)
            init_a_rows = torch.stack(
                [candidate.init_a_row.detach().float().cpu() for candidate in ordered],
                dim=0,
            ).contiguous()
            report["selected_channel_indices"] = [int(candidate.channel_index) for candidate in ordered]
            report["selected_excess_residual_energy"] = [
                float(candidate.excess_residual_energy) for candidate in ordered
            ]
            report["selected_benefit_per_parameter"] = [
                float(candidate.benefit_per_parameter) for candidate in ordered
            ]
        else:
            init_a_rows = None
            report["selected_channel_indices"] = []
            report["selected_excess_residual_energy"] = []
            report["selected_benefit_per_parameter"] = []
            if int(report.get("residual_rank", 0)) > 0:
                report.setdefault("reason", "not_selected_by_global_mdl_budget")

        selected_proposals[name] = RankGrowthProposal(
            report=report,
            init_a_rows=init_a_rows,
            channel_candidates=proposal.channel_candidates,
        )

    budget_report: dict[str, object] = {
        "effective_observations": int(effective_observations),
        "mdl_channel_budget": int(mdl_channel_budget),
        "mdl_budget_kind": "log_evidence_with_distributed_floor",
        "mdl_distributed_floor": int(
            distributed_rank_channel_floor(
                effective_observations,
                candidate_count=len(candidates),
            )
        ),
        "memory_channel_budget": int(memory_channel_budget),
        "global_channel_budget": int(global_budget),
        "candidate_channel_count": int(len(candidates)),
        "selected_channel_count": int(len(selected)),
        "selected_module_count": int(len(selected_by_module)),
        "estimated_added_parameter_count": int(sum(candidate.parameter_cost for candidate in selected)),
        "estimated_added_optimizer_bytes": int(used_memory_bytes),
    }
    if selected:
        budget_report["min_selected_benefit_per_parameter"] = float(
            min(candidate.benefit_per_parameter for candidate in selected)
        )
        budget_report["max_selected_benefit_per_parameter"] = float(
            max(candidate.benefit_per_parameter for candidate in selected)
        )
    else:
        budget_report["min_selected_benefit_per_parameter"] = 0.0
        budget_report["max_selected_benefit_per_parameter"] = 0.0

    # Include the heldout score scale so downstream analysis can compare
    # structural growth events against the validation objective being optimized.
    budget_report["baseline_validation_loss_mean"] = float(baseline_report.loss_mean)
    budget_report["baseline_validation_loss_max_snr_bucket"] = float(
        baseline_report.loss_max_snr_bucket
    )
    return selected_proposals, budget_report


def validation_effective_observations(validation_skeleton: ValidationSkeleton) -> int:
    total = 0
    for item in validation_skeleton.items:
        latent_shape = tuple(int(value) for value in item.record.latent_shape)
        if not latent_shape:
            continue
        element_count = 1
        for value in latent_shape:
            element_count *= max(1, int(value))
        total += element_count
    return max(1, int(total))


def mdl_rank_channel_budget(
    effective_observations: int,
    *,
    candidate_count: int = 0,
) -> int:
    """Return a structural budget for new rank channels.

    The base MDL budget grows logarithmically with effective validation
    observations.  To handle genuinely distributed residual signal, we also use
    a sublinear distributed floor:

        sqrt(candidate_count) * log2(log2(n) + 1).

    This is still far below all-modules-per-plateau growth, but it is large
    enough to cover a delocalized residual when many layers carry positive
    evidence.  Memory accounting remains the final hard cap.
    """
    n = max(2, int(effective_observations))
    base_budget = max(1, int(math.log2(n)))
    return max(
        base_budget,
        distributed_rank_channel_floor(
            effective_observations,
            candidate_count=candidate_count,
        ),
    )


def distributed_rank_channel_floor(
    effective_observations: int,
    *,
    candidate_count: int,
) -> int:
    if candidate_count <= 0:
        return 0
    n = max(2, int(effective_observations))
    evidence_factor = math.log2(max(2.0, math.log2(n) + 1.0))
    floor = int(math.ceil(math.sqrt(float(candidate_count)) * evidence_factor))
    return min(
        int(candidate_count),
        max(RANK_GROWTH_MIN_DISTRIBUTED_BUDGET, floor),
    )


def memory_rank_channel_budget(
    *,
    candidates: list[RankChannelCandidate],
    free_cuda_bytes: int | None,
) -> int:
    if not candidates:
        return 0
    if free_cuda_bytes is None:
        return len(candidates)
    available = max(0, int(free_cuda_bytes) - RANK_GROWTH_MIN_FREE_CUDA_BYTES)
    if available <= 0:
        return 0
    ordered_costs = sorted(
        int(candidate.parameter_cost * RANK_GROWTH_BYTES_PER_TRAINABLE_PARAMETER)
        for candidate in candidates
    )
    count = 0
    used = 0
    for cost in ordered_costs:
        if used + cost > available:
            break
        used += cost
        count += 1
    return count


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
    current_rank = int(lora_a.out_features)
    parameter_cost = int(lora_a.in_features + lora_b.out_features)

    channel_candidates: tuple[RankChannelCandidate, ...] = ()
    if residual_rank > 0 and parameter_cost > 0:
        try:
            _u, _s, vh = torch.linalg.svd(signal, full_matrices=False)
            candidate_count = min(residual_rank, RANK_GROWTH_MAX_CHANNELS, int(vh.shape[0]))
            candidates: list[RankChannelCandidate] = []
            for index in range(candidate_count):
                singular_value = float(signal_values[index].item())
                excess_energy = max(float(singular_value * singular_value - threshold * threshold), 0.0)
                if excess_energy <= 0.0:
                    continue
                candidates.append(
                    RankChannelCandidate(
                        module_name=name,
                        adapter=adapter,
                        channel_index=index,
                        singular_value=singular_value,
                        noise_threshold=float(threshold),
                        excess_residual_energy=float(excess_energy),
                        parameter_cost=parameter_cost,
                        benefit_per_parameter=float(excess_energy / max(1, parameter_cost)),
                        init_a_row=vh[index].detach().float().cpu().contiguous(),
                    )
                )
            channel_candidates = tuple(candidates)
        except RuntimeError:
            channel_candidates = ()

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
            "rank_growth": 0,
            "rank_growth_candidate_count": int(len(channel_candidates)),
            "rank_probe_kind": "dense_base_validation_gradient_mdl_budgeted",
            "parameter_cost_per_channel": int(parameter_cost),
        },
        init_a_rows=None,
        channel_candidates=channel_candidates,
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
            usable = min(int(growth), rows.shape[0])
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
