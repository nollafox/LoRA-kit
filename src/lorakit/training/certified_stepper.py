"""Context-probe instrumentation for certifying training steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ContextProbe:
    context_ids: tuple[int, ...]
    losses: torch.Tensor
    gradient_dot: torch.Tensor
    gradient_norms: torch.Tensor
    cosine: torch.Tensor
    mgda_lambda: torch.Tensor
    pareto_gap: float
    dual_thickness: float
    negative_cosine_fraction: float
    min_pairwise_cosine: float


@dataclass(frozen=True)
class StepCertificate:
    accepted: bool
    reason: str
    old_mean_loss: float
    new_mean_loss: float
    old_max_loss: float
    new_max_loss: float
    mean_delta: float
    max_delta: float
    backtracks: int
    step_size: float


def denoising_loss_per_example(
    *,
    model_pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return one denoising loss scalar per batch item."""
    if model_pred.shape != target.shape:
        raise ValueError("model_pred and target shapes must match")
    if model_pred.ndim < 2:
        raise ValueError("model_pred and target must include a batch dimension")
    loss = F.mse_loss(
        model_pred.float(),
        target.float(),
        reduction="none",
    )
    return loss.mean(dim=tuple(range(1, loss.ndim)))


def context_losses_from_examples(
    *,
    per_example_loss: torch.Tensor,
    timesteps: torch.Tensor,
) -> tuple[tuple[int, ...], torch.Tensor]:
    """Group per-example losses by exact observed diffusion timestep."""
    if per_example_loss.ndim != 1:
        raise ValueError("per_example_loss must be rank-1")
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be rank-1")
    if per_example_loss.shape[0] != timesteps.shape[0]:
        raise ValueError("loss/timestep batch mismatch")

    ids = sorted({int(timestep.item()) for timestep in timesteps.detach().cpu()})
    grouped: list[torch.Tensor] = []
    for context_id in ids:
        mask = timesteps == context_id
        if torch.any(mask):
            grouped.append(per_example_loss[mask].mean())
    if not grouped:
        raise ValueError("No timestep contexts found")
    return tuple(ids), torch.stack(grouped)


def trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def flatten_current_grads(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for parameter in parameters:
        if parameter.grad is None:
            pieces.append(torch.zeros_like(parameter, dtype=torch.float32).reshape(-1).cpu())
        else:
            pieces.append(parameter.grad.detach().float().reshape(-1).cpu())
    if not pieces:
        raise ValueError("No trainable parameters found")
    return torch.cat(pieces, dim=0)


def zero_parameter_grads(parameters: Sequence[torch.nn.Parameter]) -> None:
    for parameter in parameters:
        parameter.grad = None


def context_gradients(
    *,
    context_losses: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    retain_graph: bool = True,
) -> list[torch.Tensor]:
    """Compute one flattened trainable-parameter gradient per context loss."""
    if context_losses.ndim != 1:
        raise ValueError("context_losses must be rank-1")
    gradients: list[torch.Tensor] = []
    for index, loss in enumerate(context_losses):
        zero_parameter_grads(parameters)
        loss.backward(retain_graph=retain_graph or index < len(context_losses) - 1)
        gradients.append(flatten_current_grads(parameters))
    zero_parameter_grads(parameters)
    return gradients


def project_to_simplex(vector: torch.Tensor) -> torch.Tensor:
    """Project a rank-1 tensor onto the probability simplex."""
    if vector.ndim != 1:
        raise ValueError("vector must be rank-1")

    values = vector.detach().float()
    count = values.numel()
    if count == 0:
        raise ValueError("vector cannot be empty")
    if count == 1:
        return torch.ones_like(values)

    sorted_values, _ = torch.sort(values, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0) - 1.0
    positions = torch.arange(1, count + 1, device=values.device, dtype=values.dtype)
    active = sorted_values - cumulative / positions > 0
    if not torch.any(active):
        return torch.full_like(values, 1.0 / count)

    rho = torch.nonzero(active, as_tuple=False)[-1].item()
    theta = cumulative[rho] / float(rho + 1)
    projected = torch.clamp(values - theta, min=0.0)
    total = projected.sum()
    if total <= 0:
        return torch.full_like(values, 1.0 / count)
    return projected / total


def solve_mgda_weights(
    gradient_dot: torch.Tensor,
    *,
    max_iter: int = 250,
    tolerance: float = 1e-10,
) -> torch.Tensor:
    """Solve the MGDA minimum-norm simplex weights from a gradient Gram matrix."""
    gram = gradient_dot.detach().float().cpu()
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gradient_dot must be square")

    context_count = gram.shape[0]
    if context_count == 0:
        raise ValueError("gradient_dot cannot be empty")
    if context_count == 1:
        return torch.ones(1, dtype=torch.float32)

    gram = 0.5 * (gram + gram.T)
    gram = gram + torch.eye(context_count, dtype=gram.dtype) * 1e-12
    weights = torch.full((context_count,), 1.0 / context_count, dtype=torch.float32)
    lipschitz = torch.linalg.norm(gram, ord=2).item()
    step_size = 1.0 / max(lipschitz, 1e-12)
    previous_objective: float | None = None

    for _ in range(max_iter):
        gradient = gram @ weights
        candidate = project_to_simplex(weights - step_size * gradient)
        objective = 0.5 * float(candidate @ gram @ candidate)
        if previous_objective is not None and abs(previous_objective - objective) <= tolerance:
            weights = candidate
            break
        weights = candidate
        previous_objective = objective

    return project_to_simplex(weights)


def gradient_gram(gradients: Sequence[torch.Tensor]) -> torch.Tensor:
    if not gradients:
        raise ValueError("gradients cannot be empty")
    stacked = torch.stack([gradient.detach().float().cpu() for gradient in gradients], dim=0)
    return stacked @ stacked.T


def cosine_from_gram(gram: torch.Tensor) -> torch.Tensor:
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be square")
    norms = torch.sqrt(torch.clamp(torch.diag(gram), min=0.0))
    denominator = torch.outer(norms, norms).clamp_min(1e-12)
    cosine = gram / denominator
    return torch.clamp(cosine, min=-1.0, max=1.0)


def probe_from_context_losses(
    *,
    context_ids: tuple[int, ...],
    context_losses: torch.Tensor,
    gradients: Sequence[torch.Tensor],
) -> ContextProbe:
    if len(context_ids) != len(gradients):
        raise ValueError("context ID / gradient count mismatch")
    if context_losses.shape != (len(context_ids),):
        raise ValueError("context loss shape mismatch")

    gram = gradient_gram(gradients)
    cosine = cosine_from_gram(gram)
    weights = solve_mgda_weights(gram)
    mgda_squared_norm = float(weights @ gram @ weights)
    pareto_gap = float(max(mgda_squared_norm, 0.0) ** 0.5)
    weight_norm_squared = float(weights @ weights)
    dual_thickness = float(1.0 / max(weight_norm_squared, 1e-12))

    context_count = len(context_ids)
    if context_count > 1:
        off_diagonal = cosine[~torch.eye(context_count, dtype=torch.bool)]
        negative_cosine_fraction = float((off_diagonal < 0).float().mean().item())
        min_pairwise_cosine = float(off_diagonal.min().item())
    else:
        negative_cosine_fraction = 0.0
        min_pairwise_cosine = 1.0

    return ContextProbe(
        context_ids=context_ids,
        losses=context_losses.detach().float().cpu(),
        gradient_dot=gram.detach().float().cpu(),
        gradient_norms=torch.sqrt(torch.clamp(torch.diag(gram), min=0.0)).cpu(),
        cosine=cosine.detach().float().cpu(),
        mgda_lambda=weights.detach().float().cpu(),
        pareto_gap=pareto_gap,
        dual_thickness=dual_thickness,
        negative_cosine_fraction=negative_cosine_fraction,
        min_pairwise_cosine=min_pairwise_cosine,
    )


def probe_to_log_dict(probe: ContextProbe, *, step: int) -> dict[str, object]:
    return {
        "step": int(step),
        "context_ids": [int(context_id) for context_id in probe.context_ids],
        "context_losses": [float(value) for value in probe.losses.tolist()],
        "loss_mean": float(probe.losses.mean().item()),
        "loss_max": float(probe.losses.max().item()),
        "gradient_norms": [float(value) for value in probe.gradient_norms.tolist()],
        "dual_lambda": [float(value) for value in probe.mgda_lambda.tolist()],
        "dual_thickness": float(probe.dual_thickness),
        "pareto_gap": float(probe.pareto_gap),
        "negative_cosine_fraction": float(probe.negative_cosine_fraction),
        "min_pairwise_cosine": float(probe.min_pairwise_cosine),
    }


def weighted_context_loss(
    *,
    context_losses: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if context_losses.ndim != 1:
        raise ValueError("context_losses must be rank-1")
    if weights.shape != context_losses.shape:
        raise ValueError("weights/context loss shape mismatch")
    return torch.sum(weights.to(device=context_losses.device, dtype=context_losses.dtype) * context_losses)


def certify_losses(
    *,
    old_context_losses: torch.Tensor,
    new_context_losses: torch.Tensor,
    backtracks: int,
    step_size: float,
    tolerance: float = 1e-8,
) -> StepCertificate:
    old_losses = old_context_losses.detach().float().cpu()
    new_losses = new_context_losses.detach().float().cpu()
    if old_losses.shape != new_losses.shape:
        raise ValueError("old/new context loss shape mismatch")

    old_mean = float(old_losses.mean().item())
    new_mean = float(new_losses.mean().item())
    old_max = float(old_losses.max().item())
    new_max = float(new_losses.max().item())
    mean_delta = new_mean - old_mean
    max_delta = new_max - old_max
    accepted = max_delta <= tolerance and mean_delta <= tolerance

    return StepCertificate(
        accepted=accepted,
        reason="accepted" if accepted else "rejected_context_or_mean_increase",
        old_mean_loss=old_mean,
        new_mean_loss=new_mean,
        old_max_loss=old_max,
        new_max_loss=new_max,
        mean_delta=mean_delta,
        max_delta=max_delta,
        backtracks=backtracks,
        step_size=float(step_size),
    )
