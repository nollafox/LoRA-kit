"""Denoising loss helpers shared by training, validation, and probes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import DDPMScheduler, UNet2DConditionModel

from lorakit.errors import LorakitError
from lorakit.training.backends._policy import ObjectivePolicy, snr_loss_weights
from lorakit.training.certified_stepper import denoising_loss_per_example


@dataclass(frozen=True)
class LossContext:
    loss: torch.Tensor
    noise: torch.Tensor
    timesteps: torch.Tensor


def batch_tensor(batch: dict[str, object], key: str) -> torch.Tensor:
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise LorakitError(f"Batch value must be a tensor: {key}")
    return value


def loss_context(
    *,
    batch: dict[str, object],
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    objective: ObjectivePolicy,
) -> LossContext:
    latents = batch_tensor(batch, "latents").to(device=unet.device, dtype=weight_dtype)
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=latents.device,
    ).long()
    return loss_context_fixed(
        batch=batch,
        unet=unet,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
        objective=objective,
        noise=noise,
        timesteps=timesteps,
    )


def loss_context_fixed(
    *,
    batch: dict[str, object],
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    objective: ObjectivePolicy,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> LossContext:
    latents = batch_tensor(batch, "latents").to(device=unet.device, dtype=weight_dtype)
    encoder_hidden_states = batch_tensor(batch, "encoder_hidden_states").to(device=unet.device, dtype=weight_dtype)
    local_noise = noise.to(device=unet.device, dtype=weight_dtype)
    local_timesteps = timesteps.to(device=unet.device).long()
    noisy_latents = noise_scheduler.add_noise(latents, local_noise, local_timesteps)
    target_value = target(
        noise_scheduler=noise_scheduler,
        latents=latents,
        noise=local_noise,
        timesteps=local_timesteps,
    )
    prediction = unet(noisy_latents, local_timesteps, encoder_hidden_states, return_dict=False)[0]
    per_example_loss = denoising_loss_per_example(model_pred=prediction, target=target_value)
    loss = objective_loss(
        per_example_loss=per_example_loss,
        timesteps=local_timesteps,
        noise_scheduler=noise_scheduler,
        objective=objective,
    )
    return LossContext(
        loss=loss,
        noise=local_noise.detach().cpu(),
        timesteps=local_timesteps.detach().cpu(),
    )


def objective_loss(
    *,
    per_example_loss: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: DDPMScheduler,
    objective: ObjectivePolicy,
) -> torch.Tensor:
    if objective.name == "base_mse":
        return per_example_loss.mean()
    if objective.name == "minsnr":
        if objective.gamma is None:
            raise LorakitError("Min-SNR objective requires gamma")
        weights = snr_loss_weights(objective=objective, noise_scheduler=noise_scheduler, timesteps=timesteps)
        return torch.mean(per_example_loss * weights.to(device=per_example_loss.device, dtype=per_example_loss.dtype))
    raise LorakitError(f"Unknown training objective: {objective.name}")


def target(
    *,
    noise_scheduler: DDPMScheduler,
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    prediction_type = str(noise_scheduler.config.prediction_type)
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        return noise_scheduler.get_velocity(latents, noise, timesteps)
    raise LorakitError(f"Unknown prediction type: {prediction_type}")


def fixed_subset_context_loss(
    *,
    batch: dict[str, object],
    mask: torch.Tensor,
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    latents = batch_tensor(batch, "latents").to(device=unet.device, dtype=weight_dtype)[mask]
    encoder_hidden_states = batch_tensor(batch, "encoder_hidden_states").to(device=unet.device, dtype=weight_dtype)[mask]
    local_noise = noise.to(device=unet.device, dtype=weight_dtype)[mask]
    local_timesteps = timesteps.to(device=unet.device).long()[mask]
    noisy_latents = noise_scheduler.add_noise(latents, local_noise, local_timesteps)
    target_value = target(
        noise_scheduler=noise_scheduler,
        latents=latents,
        noise=local_noise,
        timesteps=local_timesteps,
    )
    prediction = unet(noisy_latents, local_timesteps, encoder_hidden_states, return_dict=False)[0]
    return denoising_loss_per_example(model_pred=prediction, target=target_value).mean()
