"""Model-bound validation for Diffusers training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import DDPMScheduler, UNet2DConditionModel

from lorakit.training.backends._cache import load_tensor
from lorakit.training.backends._loss import fixed_subset_context_loss
from lorakit.training.backends._validation import (
    ValidationItem,
    ValidationReport,
    ValidationSkeleton,
    evaluate_validation_skeleton,
)


@dataclass(frozen=True)
class DiffusionValidator:
    skeleton: ValidationSkeleton
    unet: UNet2DConditionModel
    noise_scheduler: DDPMScheduler
    weight_dtype: torch.dtype

    def score(self, *, step: int) -> ValidationReport:
        return evaluate_validation_skeleton(
            skeleton=self.skeleton,
            step=step,
            loss_for_item=self.loss,
        )

    def loss(self, item: ValidationItem) -> float:
        loss = self.loss_tensor(item)
        return float(loss.detach().float().cpu().item())

    def loss_tensor(self, item: ValidationItem) -> torch.Tensor:
        latent = load_tensor(item.record.latent_path, expected_shape=item.record.latent_shape)
        hidden = load_tensor(
            item.record.encoder_hidden_state_path,
            expected_shape=item.record.encoder_hidden_state_shape,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(item.noise_seed)
        noise = torch.randn(item.record.latent_shape, generator=generator, dtype=torch.float32)
        batch = {
            "latents": latent.unsqueeze(0),
            "encoder_hidden_states": hidden.unsqueeze(0),
        }
        return fixed_subset_context_loss(
            batch=batch,
            mask=torch.tensor([True], device=self.unet.device),
            unet=self.unet,
            noise_scheduler=self.noise_scheduler,
            weight_dtype=self.weight_dtype,
            noise=noise.unsqueeze(0),
            timesteps=torch.tensor([item.timestep], dtype=torch.long),
        )
