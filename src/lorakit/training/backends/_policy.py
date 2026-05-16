"""Training objective and LoRA+ policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from diffusers import DDPMScheduler

from lorakit.errors import LorakitError
from lorakit.training.backends._validation import (
    SnrBucket,
    ValidationReport,
    ValidationSkeleton,
    snr_for_timesteps,
)


class ObjectiveKind(StrEnum):
    BASE_MSE = "base_mse"
    MIN_SNR = "minsnr"


@dataclass(frozen=True)
class ObjectivePolicy:
    kind: ObjectiveKind
    gamma: float | None = None

    def __post_init__(self) -> None:
        if self.kind == ObjectiveKind.BASE_MSE and self.gamma is not None:
            raise LorakitError("Base MSE objective must not set gamma")
        if self.kind == ObjectiveKind.MIN_SNR:
            if self.gamma is None:
                raise LorakitError("Min-SNR objective requires gamma")
            if self.gamma <= 0.0:
                raise LorakitError(f"Min-SNR gamma must be positive: {self.gamma}")

    @classmethod
    def base_mse(cls) -> "ObjectivePolicy":
        return cls(kind=ObjectiveKind.BASE_MSE)

    @classmethod
    def min_snr(cls, gamma: float) -> "ObjectivePolicy":
        if gamma <= 0.0:
            raise LorakitError(f"Min-SNR gamma must be positive: {gamma}")
        return cls(kind=ObjectiveKind.MIN_SNR, gamma=float(gamma))

    @property
    def label(self) -> str:
        if self.kind == ObjectiveKind.BASE_MSE:
            return "base_mse"
        if self.kind == ObjectiveKind.MIN_SNR and self.gamma is not None:
            return f"minsnr_gamma_{self.gamma:.6g}"
        raise LorakitError(f"Cannot label objective: {self}")

    @property
    def is_min_snr(self) -> bool:
        return self.kind == ObjectiveKind.MIN_SNR


@dataclass(frozen=True)
class TrainingPolicy:
    objective: ObjectivePolicy
    lora_plus_ratio: float

    def __post_init__(self) -> None:
        if self.lora_plus_ratio <= 0.0:
            raise LorakitError(f"LoRA+ ratio must be positive: {self.lora_plus_ratio}")

    @classmethod
    def default(cls) -> "TrainingPolicy":
        return cls(objective=ObjectivePolicy.base_mse(), lora_plus_ratio=1.0)


def default_training_policy() -> TrainingPolicy:
    return TrainingPolicy.default()


def objective_candidates(
    *,
    current_policy: TrainingPolicy,
    baseline_report: ValidationReport,
    validation_skeleton: ValidationSkeleton,
    noise_scheduler: DDPMScheduler,
) -> list[ObjectivePolicy]:
    candidates = [ObjectivePolicy.base_mse()]
    gammas = [
        validation_bucket_gamma(validation_skeleton=validation_skeleton, bucket=SnrBucket.MID),
        5.0,
        validation_bucket_gamma(
            validation_skeleton=validation_skeleton,
            bucket=baseline_report.worst_bucket,
        ),
    ]
    if current_policy.objective.is_min_snr and current_policy.objective.gamma is not None:
        gammas.append(current_policy.objective.gamma)
    seen: set[float] = set()
    for gamma in gammas:
        key = round(float(gamma), 8)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(ObjectivePolicy.min_snr(float(gamma)))
    return candidates


def validation_bucket_gamma(*, validation_skeleton: ValidationSkeleton, bucket: SnrBucket) -> float:
    values = [item.snr for item in validation_skeleton.items if item.snr_bucket == bucket]
    if not values:
        raise LorakitError(f"Validation skeleton has no SNR bucket: {bucket.value}")
    return float(sum(values) / len(values))


def ratio_label(ratio: float) -> str:
    return f"ratio_{ratio:.6g}"


def lora_plus_candidate_ratios(unet: torch.nn.Module) -> list[float]:
    scale_a, scale_b = lora_relative_gradient_scales(unet)
    if scale_a <= 0.0 or scale_b <= 0.0:
        return [1.0]
    ratio = max(1e-3, min(1e3, scale_a / max(scale_b, 1e-12)))
    inverse = max(1e-3, min(1e3, 1.0 / ratio))
    candidates = [1.0, ratio, inverse]
    deduped: list[float] = []
    for value in candidates:
        if not any(abs(value - existing) <= 1e-9 for existing in deduped):
            deduped.append(float(value))
    return deduped


def lora_relative_gradient_scales(unet: torch.nn.Module) -> tuple[float, float]:
    lora_a, lora_b = partition_lora_parameters(unet)
    return relative_gradient_scale(lora_a), relative_gradient_scale(lora_b)


def partition_lora_parameters(unet: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    lora_a: list[torch.nn.Parameter] = []
    lora_b: list[torch.nn.Parameter] = []
    for name, parameter in unet.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_A" in name:
            lora_a.append(parameter)
        elif "lora_B" in name:
            lora_b.append(parameter)
    return lora_a, lora_b


def relative_gradient_scale(parameters: list[torch.nn.Parameter]) -> float:
    if not parameters:
        return 0.0
    grad_square_sum = 0.0
    value_square_sum = 0.0
    count = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_square_sum += float(torch.sum(parameter.grad.detach().float().pow(2)).item())
        value_square_sum += float(torch.sum(parameter.detach().float().pow(2)).item())
        count += int(parameter.numel())
    if count <= 0:
        return 0.0
    grad_rms = (grad_square_sum / float(count)) ** 0.5
    value_rms = (value_square_sum / float(count)) ** 0.5
    return float(grad_rms / max(value_rms, 1e-12))


def snr_loss_weights(
    *,
    objective: ObjectivePolicy,
    noise_scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    if not objective.is_min_snr:
        return torch.ones_like(timesteps, dtype=torch.float32)
    if objective.gamma is None:
        raise LorakitError("Min-SNR objective requires gamma")
    snr = snr_for_timesteps(noise_scheduler=noise_scheduler, timesteps=timesteps).clamp_min(1e-12)
    gamma = torch.tensor(float(objective.gamma), device=snr.device, dtype=snr.dtype)
    clipped = torch.minimum(snr, gamma)
    prediction_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        return clipped / snr
    if prediction_type == "v_prediction":
        return clipped / (snr + 1.0)
    raise LorakitError(f"Unknown prediction type: {prediction_type}")
