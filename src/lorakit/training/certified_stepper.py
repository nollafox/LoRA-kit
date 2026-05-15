"""Context-probe instrumentation for certifying training steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


DEFAULT_CERTIFICATE_ABSOLUTE_FLOOR = 1e-7
DEFAULT_CERTIFICATE_RELATIVE_TOLERANCE = 1e-5
DEFAULT_MGDA_MAX_ITERATIONS = 250
DEFAULT_MGDA_TOLERANCE = 1e-10
LAMBDA_SUPPORT_TOLERANCE = 1e-6
NUMERICAL_EPSILON = 1e-12
SIMPLEX_TOTAL = 1.0
ACCEPTANCE_LEVEL_RANKS = {
    "reject": 0,
    "bottleneck": 1,
    "strong": 2,
}


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

    # Dual/context instrumentation.
    context_count: int
    lambda_support: int
    lambda_entropy_bits: float
    lambda_max: float
    lambda_min_positive: float
    lambda_weighted_loss: float
    loss_min: float
    loss_range: float
    max_loss_context_id: int
    max_lambda_context_id: int
    pair_count: int
    negative_pair_count: int
    mean_pairwise_cosine: float
    max_pairwise_cosine: float
    most_conflicting_pair: tuple[int, int] | None


@dataclass(frozen=True)
class StepCertificate:
    accepted: bool
    accepted_strict: bool
    accepted_tolerant: bool
    accepted_strong: bool
    accepted_bottleneck: bool
    acceptance_level: str
    reason: str
    old_mean_loss: float
    new_mean_loss: float
    old_max_loss: float
    new_max_loss: float
    mean_delta: float
    max_delta: float
    backtracks: int
    step_size: float
    certificate_tolerance: float
    context_deltas: tuple[float, ...]
    worsened_context_ids: tuple[int, ...]


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
    cumulative = torch.cumsum(sorted_values, dim=0) - SIMPLEX_TOTAL
    positions = torch.arange(1, count + 1, device=values.device, dtype=values.dtype)
    active = sorted_values - cumulative / positions > 0
    if not torch.any(active):
        return torch.full_like(values, SIMPLEX_TOTAL / count)

    rho = torch.nonzero(active, as_tuple=False)[-1].item()
    theta = cumulative[rho] / float(rho + 1)
    projected = torch.clamp(values - theta, min=0.0)
    total = projected.sum()
    if total <= 0:
        return torch.full_like(values, SIMPLEX_TOTAL / count)
    return projected / total


def solve_mgda_weights(
    gradient_dot: torch.Tensor,
    *,
    max_iter: int = DEFAULT_MGDA_MAX_ITERATIONS,
    tolerance: float = DEFAULT_MGDA_TOLERANCE,
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
    gram = gram + torch.eye(context_count, dtype=gram.dtype) * NUMERICAL_EPSILON
    weights = torch.full((context_count,), SIMPLEX_TOTAL / context_count, dtype=torch.float32)
    lipschitz = torch.linalg.norm(gram, ord=2).item()
    step_size = SIMPLEX_TOTAL / max(lipschitz, NUMERICAL_EPSILON)
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
    denominator = torch.outer(norms, norms).clamp_min(NUMERICAL_EPSILON)
    cosine = gram / denominator
    return torch.clamp(cosine, min=-1.0, max=1.0)


def _lambda_entropy_bits(weights: torch.Tensor) -> float:
    positive = weights.detach().float().cpu()
    positive = positive[positive > 0]
    if positive.numel() == 0:
        return 0.0
    return float(-(positive * torch.log2(positive)).sum().item())


def _lambda_min_positive(weights: torch.Tensor) -> float:
    positive = weights.detach().float().cpu()
    positive = positive[positive > 0]
    if positive.numel() == 0:
        return 0.0
    return float(positive.min().item())


def _pairwise_cosine_summary(
    *,
    context_ids: tuple[int, ...],
    cosine: torch.Tensor,
) -> tuple[int, int, float, float, float, tuple[int, int] | None]:
    context_count = len(context_ids)
    if context_count <= 1:
        return 0, 0, 1.0, 1.0, 1.0, None

    indices = torch.triu_indices(context_count, context_count, offset=1)
    values = cosine[indices[0], indices[1]].detach().float().cpu()

    pair_count = int(values.numel())
    negative_pair_count = int((values < 0).sum().item())

    min_index = int(torch.argmin(values).item())
    left = int(indices[0, min_index].item())
    right = int(indices[1, min_index].item())

    return (
        pair_count,
        negative_pair_count,
        float(values.mean().item()),
        float(values.max().item()),
        float(values.min().item()),
        (int(context_ids[left]), int(context_ids[right])),
    )


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
    dual_thickness = float(SIMPLEX_TOTAL / max(weight_norm_squared, NUMERICAL_EPSILON))

    (
        pair_count,
        negative_pair_count,
        mean_pairwise_cosine,
        max_pairwise_cosine,
        min_pairwise_cosine,
        most_conflicting_pair,
    ) = _pairwise_cosine_summary(
        context_ids=context_ids,
        cosine=cosine,
    )

    negative_cosine_fraction = (
        float(negative_pair_count / pair_count)
        if pair_count > 0
        else 0.0
    )

    losses = context_losses.detach().float().cpu()
    weights_cpu = weights.detach().float().cpu()

    max_loss_index = int(torch.argmax(losses).item())
    max_lambda_index = int(torch.argmax(weights_cpu).item())

    lambda_support = int((weights_cpu > LAMBDA_SUPPORT_TOLERANCE).sum().item())

    return ContextProbe(
        context_ids=context_ids,
        losses=losses,
        gradient_dot=gram.detach().float().cpu(),
        gradient_norms=torch.sqrt(torch.clamp(torch.diag(gram), min=0.0)).cpu(),
        cosine=cosine.detach().float().cpu(),
        mgda_lambda=weights_cpu,
        pareto_gap=pareto_gap,
        dual_thickness=dual_thickness,
        negative_cosine_fraction=negative_cosine_fraction,
        min_pairwise_cosine=min_pairwise_cosine,
        context_count=len(context_ids),
        lambda_support=lambda_support,
        lambda_entropy_bits=_lambda_entropy_bits(weights_cpu),
        lambda_max=float(weights_cpu.max().item()),
        lambda_min_positive=_lambda_min_positive(weights_cpu),
        lambda_weighted_loss=float((weights_cpu * losses).sum().item()),
        loss_min=float(losses.min().item()),
        loss_range=float((losses.max() - losses.min()).item()),
        max_loss_context_id=int(context_ids[max_loss_index]),
        max_lambda_context_id=int(context_ids[max_lambda_index]),
        pair_count=pair_count,
        negative_pair_count=negative_pair_count,
        mean_pairwise_cosine=mean_pairwise_cosine,
        max_pairwise_cosine=max_pairwise_cosine,
        most_conflicting_pair=most_conflicting_pair,
    )


def probe_to_log_dict(
    probe: ContextProbe,
    *,
    step: int,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "step": int(step),
        "context_ids": [int(context_id) for context_id in probe.context_ids],
        "context_count": int(probe.context_count),
        "context_losses": [float(value) for value in probe.losses.tolist()],
        "loss_mean": float(probe.losses.mean().item()),
        "loss_max": float(probe.losses.max().item()),
        "loss_min": float(probe.loss_min),
        "loss_range": float(probe.loss_range),
        "lambda_weighted_loss": float(probe.lambda_weighted_loss),

        "gradient_norms": [float(value) for value in probe.gradient_norms.tolist()],
        "dual_lambda": [float(value) for value in probe.mgda_lambda.tolist()],
        "dual_thickness": float(probe.dual_thickness),
        "lambda_support": int(probe.lambda_support),
        "lambda_entropy_bits": float(probe.lambda_entropy_bits),
        "lambda_max": float(probe.lambda_max),
        "lambda_min_positive": float(probe.lambda_min_positive),

        "pareto_gap": float(probe.pareto_gap),

        "pair_count": int(probe.pair_count),
        "negative_pair_count": int(probe.negative_pair_count),
        "negative_cosine_fraction": float(probe.negative_cosine_fraction),
        "min_pairwise_cosine": float(probe.min_pairwise_cosine),
        "mean_pairwise_cosine": float(probe.mean_pairwise_cosine),
        "max_pairwise_cosine": float(probe.max_pairwise_cosine),
        "most_conflicting_pair": (
            [int(probe.most_conflicting_pair[0]), int(probe.most_conflicting_pair[1])]
            if probe.most_conflicting_pair is not None
            else None
        ),

        "max_loss_context_id": int(probe.max_loss_context_id),
        "max_lambda_context_id": int(probe.max_lambda_context_id),
    }

    if extra:
        payload.update(extra)

    return payload



def _normalize_weights(values: torch.Tensor) -> torch.Tensor:
    weights = values.detach().float().cpu().reshape(-1)
    if weights.numel() == 0:
        raise ValueError("weights cannot be empty")
    if torch.any(weights < 0):
        raise ValueError("weights must be nonnegative")
    total = float(weights.sum().item())
    if total <= 0:
        return torch.full_like(weights, SIMPLEX_TOTAL / weights.numel())
    return weights / total


def normalized_count_weights(context_counts: Sequence[int]) -> torch.Tensor:
    if not context_counts:
        raise ValueError("context_counts cannot be empty")
    counts = torch.tensor([int(count) for count in context_counts], dtype=torch.float32)
    if torch.any(counts <= 0):
        raise ValueError("context_counts must be positive")
    return _normalize_weights(counts)


def uniform_context_weights(context_count: int) -> torch.Tensor:
    if context_count <= 0:
        raise ValueError("context_count must be positive")
    return torch.full((context_count,), SIMPLEX_TOTAL / context_count, dtype=torch.float32)


def weighted_gradient(
    *,
    gradients: Sequence[torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    if not gradients:
        raise ValueError("gradients cannot be empty")
    weights_cpu = _normalize_weights(weights)
    if weights_cpu.shape != (len(gradients),):
        raise ValueError("weights/gradients shape mismatch")
    stacked = torch.stack([gradient.detach().float().cpu() for gradient in gradients], dim=0)
    return torch.sum(stacked * weights_cpu[:, None], dim=0)


def candidate_first_order_summaries(
    *,
    context_ids: tuple[int, ...],
    context_losses: torch.Tensor,
    gradients: Sequence[torch.Tensor],
    mgda_weights: torch.Tensor,
    context_counts: Sequence[int] | None = None,
    step_size: float = SIMPLEX_TOTAL,
    tolerance: float = NUMERICAL_EPSILON,
) -> dict[str, object]:
    """Summarize first-order context-loss changes for candidate descent directions.

    These are local linear predictions using the measured context gradients:
        L_i(theta - eta * g_candidate) - L_i(theta)
            ~= -eta * <g_i, g_candidate>.

    They are instrumentation only; callers that need a certificate should also
    evaluate the candidate after a virtual parameter step on a frozen probe.
    """
    if len(context_ids) != len(gradients):
        raise ValueError("context ID / gradient count mismatch")
    if context_losses.shape != (len(context_ids),):
        raise ValueError("context loss shape mismatch")

    context_count = len(context_ids)
    losses = context_losses.detach().float().cpu()
    gradient_matrix = torch.stack(
        [gradient.detach().float().cpu() for gradient in gradients],
        dim=0,
    )

    candidates: dict[str, torch.Tensor] = {
        "mean_context_sgd_proxy": uniform_context_weights(context_count),
        "mgda_sgd_proxy": _normalize_weights(mgda_weights),
    }
    if context_counts is not None:
        candidates["mean_example_sgd_proxy"] = normalized_count_weights(context_counts)

    summaries: dict[str, object] = {}
    old_max = float(losses.max().item())

    for name, weights in candidates.items():
        candidate_gradient = weighted_gradient(gradients=gradients, weights=weights)
        predicted_deltas = -float(step_size) * (gradient_matrix @ candidate_gradient)
        predicted_new_losses = losses + predicted_deltas
        worsened = [
            int(context_id)
            for context_id, delta in zip(context_ids, predicted_deltas.tolist(), strict=True)
            if float(delta) > tolerance
        ]

        summaries[name] = {
            "weights": [float(value) for value in weights.detach().float().cpu().tolist()],
            "gradient_norm": float(torch.linalg.vector_norm(candidate_gradient).item()),
            "predicted_context_deltas": [
                float(value) for value in predicted_deltas.detach().float().cpu().tolist()
            ],
            "predicted_context_losses": [
                float(value) for value in predicted_new_losses.detach().float().cpu().tolist()
            ],
            "predicted_mean_delta": float(predicted_deltas.mean().item()),
            "predicted_max_loss_delta": float(predicted_new_losses.max().item() - old_max),
            "predicted_worst_context_id": int(
                context_ids[int(torch.argmax(predicted_new_losses).item())]
            ),
            "predicted_worsened_context_ids": worsened,
            "first_order_nonworsening": len(worsened) == 0,
        }

    return summaries


def weighted_context_loss(
    *,
    context_losses: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if context_losses.ndim != 1:
        raise ValueError("context_losses must be rank-1")
    if weights.shape != context_losses.shape:
        raise ValueError("weights/context loss shape mismatch")
    normalized_weights = weights.to(device=context_losses.device, dtype=context_losses.dtype)
    return torch.sum(normalized_weights * context_losses)


def adaptive_certificate_tolerance(
    context_losses: torch.Tensor,
    *,
    absolute_floor: float = DEFAULT_CERTIFICATE_ABSOLUTE_FLOOR,
    relative: float = DEFAULT_CERTIFICATE_RELATIVE_TOLERANCE,
) -> float:
    """Return a scale-aware tolerance for frozen-probe candidate certificates."""
    losses = context_losses.detach().float().cpu()
    if losses.numel() == 0:
        raise ValueError("context_losses cannot be empty")
    scale = max(float(losses.mean().item()), float(losses.max().item()), 1.0)
    return float(max(float(absolute_floor), float(relative) * scale))


def _certificate_acceptance_level(
    *,
    accepted_strong: bool,
    accepted_bottleneck: bool,
) -> str:
    if accepted_strong:
        return "strong"
    if accepted_bottleneck:
        return "bottleneck"
    return "reject"


def certificate_to_log_dict(certificate: StepCertificate) -> dict[str, object]:
    return {
        "accepted": bool(certificate.accepted),
        "accepted_strict": bool(certificate.accepted_strict),
        "accepted_tolerant": bool(certificate.accepted_tolerant),
        "accepted_strong": bool(certificate.accepted_strong),
        "accepted_bottleneck": bool(certificate.accepted_bottleneck),
        "acceptance_level": certificate.acceptance_level,
        "reason": certificate.reason,
        "old_mean_loss": float(certificate.old_mean_loss),
        "new_mean_loss": float(certificate.new_mean_loss),
        "old_max_loss": float(certificate.old_max_loss),
        "new_max_loss": float(certificate.new_max_loss),
        "mean_delta": float(certificate.mean_delta),
        "max_delta": float(certificate.max_delta),
        "backtracks": int(certificate.backtracks),
        "step_size": float(certificate.step_size),
        "certificate_tolerance": float(certificate.certificate_tolerance),
        "context_deltas": [float(value) for value in certificate.context_deltas],
        "worsened_context_ids": [int(value) for value in certificate.worsened_context_ids],
    }


def select_preferred_candidate(
    certificates: dict[str, StepCertificate],
    *,
    update_norms: dict[str, float] | None = None,
) -> dict[str, object]:
    """Rank candidate certificates without mutating model state.

    Preference order:
        1. strong certificates
        2. bottleneck certificates
        3. lower max-delta
        4. lower mean-delta
        5. smaller update norm
    """
    if not certificates:
        return {
            "selected": None,
            "reason": "no_candidates",
            "accepted_candidates": [],
            "rejected_candidates": [],
            "would_replace_committed_update": False,
        }

    if update_norms is not None:
        missing_names = set(certificates) - set(update_norms)
        if missing_names:
            missing = ", ".join(sorted(missing_names))
            raise ValueError(f"Missing update norm for candidate(s): {missing}")

    def sort_key(item: tuple[str, StepCertificate]) -> tuple[float, float, float, float]:
        name, certificate = item
        if certificate.acceptance_level not in ACCEPTANCE_LEVEL_RANKS:
            raise ValueError(f"Unknown acceptance level: {certificate.acceptance_level}")
        update_norm = 0.0 if update_norms is None else float(update_norms[name])
        return (
            -float(ACCEPTANCE_LEVEL_RANKS[certificate.acceptance_level]),
            float(certificate.max_delta),
            float(certificate.mean_delta),
            update_norm,
        )

    ordered = sorted(certificates.items(), key=sort_key)
    selected_name, selected_cert = ordered[0]
    accepted = [
        name
        for name, cert in certificates.items()
        if cert.acceptance_level != "reject"
    ]
    rejected = [
        name
        for name, cert in certificates.items()
        if cert.acceptance_level == "reject"
    ]

    return {
        "selected": selected_name,
        "reason": (
            "best_certificate"
            if selected_cert.acceptance_level != "reject"
            else "all_candidates_rejected_lowest_damage"
        ),
        "selected_acceptance_level": selected_cert.acceptance_level,
        "accepted_candidates": accepted,
        "rejected_candidates": rejected,
        "would_replace_committed_update": selected_name != "adamw_actual",
    }


def certify_losses(
    *,
    old_context_losses: torch.Tensor,
    new_context_losses: torch.Tensor,
    backtracks: int,
    step_size: float,
    tolerance: float | None = None,
    context_ids: Sequence[int] | None = None,
) -> StepCertificate:
    old_losses = old_context_losses.detach().float().cpu()
    new_losses = new_context_losses.detach().float().cpu()
    if old_losses.shape != new_losses.shape:
        raise ValueError("old/new context loss shape mismatch")

    if tolerance is None:
        tolerance = adaptive_certificate_tolerance(old_losses)

    old_mean = float(old_losses.mean().item())
    new_mean = float(new_losses.mean().item())
    old_max = float(old_losses.max().item())
    new_max = float(new_losses.max().item())
    deltas = (new_losses - old_losses).detach().float().cpu()
    mean_delta = new_mean - old_mean
    max_delta = new_max - old_max

    accepted_strict = max_delta <= 0.0 and mean_delta <= 0.0
    accepted_tolerant = max_delta <= tolerance and mean_delta <= tolerance
    accepted_strong = accepted_tolerant and bool(torch.all(deltas <= float(tolerance)).item())
    accepted_bottleneck = accepted_tolerant
    accepted = accepted_bottleneck
    acceptance_level = _certificate_acceptance_level(
        accepted_strong=accepted_strong,
        accepted_bottleneck=accepted_bottleneck,
    )

    if context_ids is None:
        context_ids = tuple(range(int(deltas.numel())))
    if len(context_ids) != int(deltas.numel()):
        raise ValueError("context_ids length must match context losses")

    worsened = tuple(
        int(context_id)
        for context_id, delta in zip(context_ids, deltas.tolist(), strict=True)
        if float(delta) > float(tolerance)
    )

    if accepted_strong:
        reason = "accepted_strong_no_context_worsened"
    elif accepted_bottleneck:
        reason = "accepted_bottleneck_mean_and_max_nonworsening"
    else:
        reason = "rejected_context_or_mean_increase"

    return StepCertificate(
        accepted=accepted,
        accepted_strict=accepted_strict,
        accepted_tolerant=accepted_tolerant,
        accepted_strong=accepted_strong,
        accepted_bottleneck=accepted_bottleneck,
        acceptance_level=acceptance_level,
        reason=reason,
        old_mean_loss=old_mean,
        new_mean_loss=new_mean,
        old_max_loss=old_max,
        new_max_loss=new_max,
        mean_delta=mean_delta,
        max_delta=max_delta,
        backtracks=backtracks,
        step_size=float(step_size),
        certificate_tolerance=float(tolerance),
        context_deltas=tuple(float(value) for value in deltas.tolist()),
        worsened_context_ids=worsened,
    )
