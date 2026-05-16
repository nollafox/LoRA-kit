"""Probe and LoRA artifact helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft.utils import get_peft_model_state_dict

from lorakit.errors import LorakitError
from lorakit.training.backends._cache import CachedBatch
from lorakit.training.backends._validation import snr_for_timesteps


LORA_WEIGHTS_NAME = "pytorch_lora_weights.safetensors"


def probe_extra_metadata(
    *,
    batch: CachedBatch,
    probe_batches,
    noise_scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
    requested_batch_size: int,
    gradient_accumulation: int,
    training_loss: torch.Tensor,
    cuda_memory_before_probe: dict[str, int] | None,
    probe_version: int,
    cuda_memory_snapshot,
) -> dict[str, object]:
    latents = batch.latents
    encoder_hidden_states = batch.encoder_hidden_states
    detached_timesteps = timesteps.detach().cpu()
    snr = snr_for_timesteps(noise_scheduler=noise_scheduler, timesteps=detached_timesteps.long()).detach().float().cpu()

    window_timesteps: list[int] = []
    window_snr_values: list[float] = []
    window_record_indices: list[int] = []
    window_images: list[str] = []
    window_sources: list[str] = []
    window_latent_item_shapes: list[list[int]] = []
    for probe_batch in probe_batches:
        probe_latents = probe_batch.batch.latents
        probe_timesteps = probe_batch.timesteps.detach().cpu().long()
        probe_snr = snr_for_timesteps(noise_scheduler=noise_scheduler, timesteps=probe_timesteps).detach().float().cpu()
        batch_size = int(probe_latents.shape[0])
        window_timesteps.extend(int(item) for item in probe_timesteps.tolist())
        window_snr_values.extend(float(item) for item in probe_snr.tolist())
        window_record_indices.extend(jsonable_int_list(probe_batch.batch.record_indices))
        window_images.extend(jsonable_str_list(probe_batch.batch.images))
        window_sources.extend([probe_batch.source] * batch_size)
        window_latent_item_shapes.extend([[int(item) for item in probe_latents.shape[1:]]] * batch_size)

    window_snr = torch.tensor(window_snr_values, dtype=torch.float32)
    return {
        "probe_version": int(probe_version),
        "microbatch_size": int(latents.shape[0]),
        "requested_batch_size": int(requested_batch_size),
        "gradient_accumulation": int(gradient_accumulation),
        "latent_batch_shape": [int(item) for item in latents.shape],
        "latent_item_shape": [int(item) for item in latents.shape[1:]],
        "encoder_hidden_state_batch_shape": [int(item) for item in encoder_hidden_states.shape],
        "encoder_hidden_state_item_shape": [int(item) for item in encoder_hidden_states.shape[1:]],
        "record_indices": jsonable_int_list(batch.record_indices),
        "images": jsonable_str_list(batch.images),
        "timesteps": [int(item) for item in detached_timesteps.tolist()],
        "snr": [float(item) for item in snr.tolist()],
        "snr_min": float(snr.min().item()),
        "snr_max": float(snr.max().item()),
        "snr_mean": float(snr.mean().item()),
        "probe_window_size": int(len(window_timesteps)),
        "probe_window_extra_size": int(max(0, len(window_timesteps) - int(latents.shape[0]))),
        "probe_window_batch_count": int(len(probe_batches)),
        "probe_window_record_indices": window_record_indices,
        "probe_window_images": window_images,
        "probe_window_sources": window_sources,
        "probe_window_latent_item_shapes": window_latent_item_shapes,
        "probe_window_timesteps": window_timesteps,
        "probe_window_snr": window_snr_values,
        "probe_window_snr_min": float(window_snr.min().item()) if window_snr.numel() else None,
        "probe_window_snr_max": float(window_snr.max().item()) if window_snr.numel() else None,
        "probe_window_snr_mean": float(window_snr.mean().item()) if window_snr.numel() else None,
        "prediction_type": str(noise_scheduler.config.prediction_type),
        "num_train_timesteps": int(noise_scheduler.config.num_train_timesteps),
        "training_loss": float(training_loss.detach().float().cpu().item()),
        "cuda_memory_before_probe": cuda_memory_before_probe,
        "cuda_memory_after_probe": cuda_memory_snapshot(),
    }


def write_policy_challenger_report_if_main(
    *,
    probe_log_path: Path,
    step: int,
    payload: dict[str, object],
    should_log: bool,
) -> None:
    if not should_log:
        return
    record = {"step": int(step), "probe_status": "quality_challengers"}
    record.update(payload)
    with probe_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def write_probe_summary(*, probe_log_path: Path, summary_path: Path) -> None:
    probes: list[dict[str, object]] = []
    validation_reports: list[dict[str, object]] = []
    quality_reports: list[dict[str, object]] = []
    with probe_log_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict) and "candidate_selection" in payload:
                probes.append(payload)
            if isinstance(payload, dict) and payload.get("probe_status") == "validation":
                validation_reports.append(payload)
            if isinstance(payload, dict) and payload.get("probe_status") == "quality_challengers":
                quality_reports.append(payload)

    dual_thickness = [float(probe.get("dual_thickness", 0.0)) for probe in probes]
    conflict_count = sum(1 for probe in probes if float(probe.get("negative_cosine_fraction", 0.0)) > 0.0)
    guardrails = [
        probe.get("candidate_selection", {}).get("guardrail", {})
        for probe in probes
        if isinstance(probe.get("candidate_selection"), dict)
    ]
    summary = {
        "probe_count": len(probes),
        "adamw_selected": sum(1 for guardrail in guardrails if guardrail.get("committed_update") == "adamw_actual"),
        "adamw_rejected": sum(
            1
            for probe in probes
            if not probe.get("candidate_certificates", {}).get("adamw_actual", {}).get("accepted_bottleneck", False)
        ),
        "fallback_used": sum(1 for guardrail in guardrails if bool(guardrail.get("fallback_used", False))),
        "updates_skipped": sum(1 for guardrail in guardrails if bool(guardrail.get("skipped_update", False))),
        "mean_dual_thickness": float(sum(dual_thickness) / len(dual_thickness)) if dual_thickness else 0.0,
        "max_dual_thickness": float(max(dual_thickness)) if dual_thickness else 0.0,
        "conflict_probe_fraction": float(conflict_count / len(probes)) if probes else 0.0,
        "validation_count": len(validation_reports),
        "best_checkpoint_step": best_validation_step(validation_reports),
        "quality_challenger_count": len(quality_reports),
        "objective_switch_count": sum(1 for report in quality_reports if bool(report.get("objective_switched", False))),
        "lora_plus_switch_count": sum(1 for report in quality_reports if bool(report.get("lora_plus", {}).get("ratio_switched", False))),
        "rank_growth_attempts": sum(1 for report in quality_reports if report.get("rank_probe", {}).get("trigger") == "validation_plateau"),
        "rank_growth_accepts": sum(1 for report in quality_reports if bool(report.get("rank_probe", {}).get("rank_growth_accepted", False))),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def best_validation_step(validation_reports: list[dict[str, object]]) -> int | None:
    improved = [report for report in validation_reports if bool(report.get("best_checkpoint_improved", False))]
    if not improved:
        return None
    return int(improved[-1]["step"])


def save_named_lora_weights(*, unet: torch.nn.Module, output_dir: Path, filename: str) -> None:
    lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=lora_layers,
        safe_serialization=True,
    )
    default_path = output_dir / LORA_WEIGHTS_NAME
    target_path = output_dir / filename
    if filename != LORA_WEIGHTS_NAME:
        if not default_path.exists():
            raise LorakitError(f"Diffusers did not write expected LoRA weights: {default_path}")
        shutil.copy2(default_path, target_path)


def jsonable_int_list(value: object) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, list | tuple):
        return [int(item) for item in value]
    return []


def jsonable_str_list(value: object) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return []
