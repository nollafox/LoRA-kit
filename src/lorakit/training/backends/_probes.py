"""Context probing and certified update guardrails."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from diffusers import DDPMScheduler, UNet2DConditionModel

from lorakit.errors import LorakitError
from lorakit.training.backends._artifacts import probe_extra_metadata
from lorakit.training.backends._cache import CachedBatch, DiskCachedLatentDataset, collate_cached
from lorakit.training.backends._loss import fixed_subset_context_loss
from lorakit.training.backends._trial import (
    apply_flat_gradient_step,
    flatten_parameters,
    scale_update,
    trial_state,
    update_norm,
)
from lorakit.training.certified_stepper import (
    StepCertificate,
    candidate_first_order_summaries,
    certificate_to_log_dict,
    certify_losses,
    probe_from_context_losses,
    probe_to_log_dict,
    select_preferred_candidate,
    trainable_parameters,
    weighted_gradient,
)


@dataclass(frozen=True)
class ProbeConfig:
    initial_steps: int = 3
    every_steps: int = 25
    min_free_cuda_bytes: int = 500_000_000
    window_target_items: int = 4
    version: int = 5
    backtrack_factors: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)


@dataclass(frozen=True)
class ProbeBatch:
    batch: CachedBatch
    noise: torch.Tensor
    timesteps: torch.Tensor
    source: str


@dataclass(frozen=True)
class GuardrailDecision:
    handled_update: bool
    committed_update: str
    backtracks: int
    fallback_used: bool
    skipped_update: bool


def default_guardrail_decision() -> GuardrailDecision:
    return GuardrailDecision(
        handled_update=False,
        committed_update="adamw_actual",
        backtracks=0,
        fallback_used=False,
        skipped_update=False,
    )


def should_probe_contexts(
    *,
    global_step: int,
    sync_gradients: bool,
    free_cuda_bytes: int | None,
    config: ProbeConfig,
) -> bool:
    if not sync_gradients:
        return False
    if global_step < config.initial_steps:
        return True
    if global_step % config.every_steps != 0:
        return False
    return free_cuda_bytes is None or free_cuda_bytes >= config.min_free_cuda_bytes


def write_probe_status_if_main(
    *,
    probe_log_path: Path,
    step: int,
    should_log: bool,
    status: str,
    free_cuda_bytes: int | None,
    extra: dict[str, object] | None = None,
) -> None:
    if not should_log:
        return
    payload: dict[str, object] = {
        "step": int(step),
        "probe_status": status,
        "free_cuda_bytes": free_cuda_bytes,
    }
    if extra:
        payload.update(extra)
    probe_log_path.parent.mkdir(parents=True, exist_ok=True)
    with probe_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


@dataclass
class ContextProber:
    unet: UNet2DConditionModel
    optimizer: object
    dataset: DiskCachedLatentDataset
    noise_scheduler: DDPMScheduler
    weight_dtype: torch.dtype
    config: ProbeConfig
    cuda_memory_snapshot: Callable[[], dict[str, int] | None]

    def run(
        self,
        *,
        batch: CachedBatch,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        probe_log_path: Path,
        step: int,
        should_log: bool,
        requested_batch_size: int,
        gradient_accumulation: int,
        training_loss: torch.Tensor,
        candidate_step_size: float,
        cuda_memory_before_probe: dict[str, int] | None,
    ) -> GuardrailDecision:
        parameters = trainable_parameters(self.unet)
        probe_batches = self._probe_window(
            current_batch=batch,
            current_noise=noise,
            current_timesteps=timesteps,
        )
        context_ids, context_counts, context_losses, gradients = self._context_gradients(
            probe_batches=probe_batches,
            parameters=parameters,
        )
        if not gradients:
            return default_guardrail_decision()

        probe = probe_from_context_losses(
            context_ids=context_ids,
            context_losses=context_losses,
            gradients=gradients,
        )
        first_order = candidate_first_order_summaries(
            context_ids=context_ids,
            context_losses=context_losses,
            gradients=gradients,
            mgda_weights=probe.mgda_lambda,
            context_counts=context_counts,
            step_size=candidate_step_size,
        )
        candidate_gradients = candidate_gradients_from_probe(
            gradients=gradients,
            mgda_weights=probe.mgda_lambda,
            context_counts=context_counts,
        )
        certificates, selection, decision = self._candidate_certificates(
            parameters=parameters,
            probe_batches=probe_batches,
            context_ids=context_ids,
            old_context_losses=context_losses,
            candidate_gradients=candidate_gradients,
            step_size=candidate_step_size,
        )

        extra = probe_extra_metadata(
            batch=batch,
            probe_batches=probe_batches,
            noise_scheduler=self.noise_scheduler,
            timesteps=timesteps,
            requested_batch_size=requested_batch_size,
            gradient_accumulation=gradient_accumulation,
            training_loss=training_loss,
            cuda_memory_before_probe=cuda_memory_before_probe,
            probe_version=self.config.version,
            cuda_memory_snapshot=self.cuda_memory_snapshot,
        )
        extra.update(
            {
                "context_example_counts": [int(count) for count in context_counts],
                "candidate_step_size": float(candidate_step_size),
                "candidate_first_order": first_order,
                "candidate_certificates": certificates,
                "candidate_selection": selection,
            }
        )

        if should_log:
            payload = probe_to_log_dict(probe, step=step, extra=extra)
            probe_log_path.parent.mkdir(parents=True, exist_ok=True)
            with probe_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")
        return decision

    def _probe_window(
        self,
        *,
        current_batch: CachedBatch,
        current_noise: torch.Tensor,
        current_timesteps: torch.Tensor,
    ) -> list[ProbeBatch]:
        probe_batches = [
            ProbeBatch(
                batch=current_batch,
                noise=current_noise.detach().cpu(),
                timesteps=current_timesteps.detach().cpu().long(),
                source="training_batch",
            )
        ]
        current_size = current_batch.size
        extra_needed = max(0, int(self.config.window_target_items) - current_size)
        if extra_needed <= 0 or len(self.dataset) <= 0:
            return probe_batches
        excluded = set(current_batch.record_indices)
        candidates = [index for index in range(len(self.dataset)) if index not in excluded]
        random.shuffle(candidates)
        for index in candidates[:extra_needed]:
            item = self.dataset[index]
            extra_batch = collate_cached([item])
            extra_noise, extra_timesteps = self._sample_noise_and_timesteps(batch=extra_batch)
            probe_batches.append(
                ProbeBatch(
                    batch=extra_batch,
                    noise=extra_noise,
                    timesteps=extra_timesteps,
                    source="streamed_probe_item",
                )
            )
        return probe_batches

    def _sample_noise_and_timesteps(self, *, batch: CachedBatch) -> tuple[torch.Tensor, torch.Tensor]:
        latents = batch.latents.to(device=self.unet.device, dtype=self.weight_dtype)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
        ).long()
        return noise.detach().cpu(), timesteps.detach().cpu()

    def _context_gradients(
        self,
        *,
        probe_batches: list[ProbeBatch],
        parameters: list[torch.nn.Parameter],
    ) -> tuple[tuple[int, ...], list[int], torch.Tensor, list[torch.Tensor]]:
        loss_sums: dict[int, torch.Tensor] = {}
        gradient_sums: dict[int, torch.Tensor] = {}
        counts: dict[int, int] = {}

        for probe_batch in probe_batches:
            device_timesteps = probe_batch.timesteps.to(device=self.unet.device).long()
            for context_id in sorted({int(value) for value in device_timesteps.detach().cpu().tolist()}):
                mask = device_timesteps == context_id
                if not torch.any(mask):
                    continue
                example_count = int(mask.sum().item())
                context_loss = fixed_subset_context_loss(
                    batch=probe_batch.batch,
                    mask=mask,
                    unet=self.unet,
                    noise_scheduler=self.noise_scheduler,
                    weight_dtype=self.weight_dtype,
                    noise=probe_batch.noise,
                    timesteps=probe_batch.timesteps,
                )
                grads = torch.autograd.grad(
                    context_loss,
                    parameters,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                flat_gradient = flatten_grads_from_autograd(parameters=parameters, grads=grads)
                weighted_loss = context_loss.detach().float().cpu() * float(example_count)
                weighted_gradient = flat_gradient * float(example_count)
                if context_id in loss_sums:
                    loss_sums[context_id] = loss_sums[context_id] + weighted_loss
                    gradient_sums[context_id] = gradient_sums[context_id] + weighted_gradient
                    counts[context_id] += example_count
                else:
                    loss_sums[context_id] = weighted_loss
                    gradient_sums[context_id] = weighted_gradient
                    counts[context_id] = example_count
                del context_loss
                del grads
                del flat_gradient

        context_ids = tuple(sorted(loss_sums))
        if not context_ids:
            raise LorakitError("Context probe produced no timestep contexts")
        context_counts = [int(counts[context_id]) for context_id in context_ids]
        context_losses = torch.stack([
            loss_sums[context_id] / float(counts[context_id])
            for context_id in context_ids
        ])
        gradients = [
            gradient_sums[context_id] / float(counts[context_id])
            for context_id in context_ids
        ]
        return context_ids, context_counts, context_losses, gradients

    def _candidate_certificates(
        self,
        *,
        parameters: list[torch.nn.Parameter],
        probe_batches: list[ProbeBatch],
        context_ids: tuple[int, ...],
        old_context_losses: torch.Tensor,
        candidate_gradients: dict[str, torch.Tensor],
        step_size: float,
    ) -> tuple[dict[str, object], dict[str, object], GuardrailDecision]:
        baseline_trial = trial_state(parameters, self.optimizer)
        certificate_objects: dict[str, StepCertificate] = {}
        results: dict[str, object] = {}
        update_norms: dict[str, float] = {}
        backtrack_factors_by_name: dict[str, float] = {}
        guardrail = default_guardrail_decision()

        try:
            if baseline_trial.has_optimizer_state:
                self._certify_adamw(
                    baseline_trial=baseline_trial,
                    parameters=parameters,
                    probe_batches=probe_batches,
                    context_ids=context_ids,
                    old_context_losses=old_context_losses,
                    step_size=step_size,
                    certificate_objects=certificate_objects,
                    results=results,
                    update_norms=update_norms,
                    backtrack_factors_by_name=backtrack_factors_by_name,
                )
            else:
                results["adamw_actual"] = {"available": False, "reason": "optimizer_state_snapshot_unavailable"}

            self._certify_gradient_candidates(
                baseline_trial=baseline_trial,
                parameters=parameters,
                probe_batches=probe_batches,
                context_ids=context_ids,
                old_context_losses=old_context_losses,
                candidate_gradients=candidate_gradients,
                step_size=step_size,
                certificate_objects=certificate_objects,
                results=results,
                update_norms=update_norms,
            )

            selection = select_preferred_candidate(certificate_objects, update_norms=update_norms)
            guardrail = self._commit_selected_candidate(
                selection=selection,
                baseline_trial=baseline_trial,
                parameters=parameters,
                candidate_gradients=candidate_gradients,
                certificate_objects=certificate_objects,
                backtrack_factors_by_name=backtrack_factors_by_name,
                step_size=step_size,
            )
            selection["guardrail"] = {
                "enabled": True,
                "committed_update": guardrail.committed_update,
                "backtracks": int(guardrail.backtracks),
                "fallback_used": bool(guardrail.fallback_used),
                "skipped_update": bool(guardrail.skipped_update),
            }
            return results, selection, guardrail
        finally:
            if not guardrail.handled_update:
                baseline_trial.restore()

    def _certify_adamw(
        self,
        *,
        baseline_trial,
        parameters: list[torch.nn.Parameter],
        probe_batches: list[ProbeBatch],
        context_ids: tuple[int, ...],
        old_context_losses: torch.Tensor,
        step_size: float,
        certificate_objects: dict[str, StepCertificate],
        results: dict[str, object],
        update_norms: dict[str, float],
        backtrack_factors_by_name: dict[str, float],
    ) -> None:
        with trial_state(parameters, self.optimizer):
            baseline_trial.restore()
            before = flatten_parameters(parameters)
            self.optimizer.step()
            after = flatten_parameters(parameters)
            update_norms["adamw_actual"] = float(torch.linalg.vector_norm(after - before).item())
            certificate = self._certificate_for_current_state(
                probe_batches=probe_batches,
                context_ids=context_ids,
                old_context_losses=old_context_losses,
                backtracks=0,
                step_size=step_size,
            )
        certificate_objects["adamw_actual"] = certificate
        results["adamw_actual"] = certificate_to_log_dict(certificate)
        if certificate.accepted_bottleneck:
            return

        for backtracks, factor in enumerate(self.config.backtrack_factors[1:], start=1):
            with trial_state(parameters, self.optimizer):
                baseline_trial.restore()
                candidate_name = f"adamw_backtrack_{factor:g}"
                before = [parameter.detach().clone() for parameter in parameters]
                self.optimizer.step()
                scale_update(parameters=parameters, before=before, factor=factor)
                update_norms[candidate_name] = update_norm(parameters=parameters, before=before)
                backtrack_factors_by_name[candidate_name] = float(factor)
                certificate = self._certificate_for_current_state(
                    probe_batches=probe_batches,
                    context_ids=context_ids,
                    old_context_losses=old_context_losses,
                    backtracks=backtracks,
                    step_size=step_size * factor,
                )
            results[candidate_name] = certificate_to_log_dict(certificate)
            if certificate.accepted_bottleneck:
                certificate_objects[candidate_name] = certificate
                break

    def _certify_gradient_candidates(
        self,
        *,
        baseline_trial,
        parameters: list[torch.nn.Parameter],
        probe_batches: list[ProbeBatch],
        context_ids: tuple[int, ...],
        old_context_losses: torch.Tensor,
        candidate_gradients: dict[str, torch.Tensor],
        step_size: float,
        certificate_objects: dict[str, StepCertificate],
        results: dict[str, object],
        update_norms: dict[str, float],
    ) -> None:
        for name, flat_gradient in candidate_gradients.items():
            with trial_state(parameters, self.optimizer):
                baseline_trial.restore()
                apply_flat_gradient_step(parameters=parameters, flat_gradient=flat_gradient, step_size=step_size)
                update_norms[name] = float(torch.linalg.vector_norm(flat_gradient).item() * abs(float(step_size)))
                certificate = self._certificate_for_current_state(
                    probe_batches=probe_batches,
                    context_ids=context_ids,
                    old_context_losses=old_context_losses,
                    backtracks=0,
                    step_size=step_size,
                )
            certificate_objects[name] = certificate
            results[name] = certificate_to_log_dict(certificate)

    def _commit_selected_candidate(
        self,
        *,
        selection: dict[str, object],
        baseline_trial,
        parameters: list[torch.nn.Parameter],
        candidate_gradients: dict[str, torch.Tensor],
        certificate_objects: dict[str, StepCertificate],
        backtrack_factors_by_name: dict[str, float],
        step_size: float,
    ) -> GuardrailDecision:
        selected = selection.get("selected")
        if selected == "adamw_actual":
            return GuardrailDecision(
                handled_update=False,
                committed_update=str(selected),
                backtracks=int(certificate_objects[str(selected)].backtracks),
                fallback_used=False,
                skipped_update=False,
            )
        if isinstance(selected, str) and selected in backtrack_factors_by_name:
            if not baseline_trial.has_optimizer_state:
                raise LorakitError(f"Cannot commit backtracked optimizer candidate without optimizer state: {selected}")
            baseline_trial.restore()
            with trial_state(parameters, self.optimizer) as trial:
                before = [parameter.detach().clone() for parameter in parameters]
                self.optimizer.step()
                scale_update(parameters=parameters, before=before, factor=backtrack_factors_by_name[selected])
                trial.commit()
            return GuardrailDecision(
                handled_update=True,
                committed_update=selected,
                backtracks=int(certificate_objects[selected].backtracks),
                fallback_used=False,
                skipped_update=False,
            )
        if isinstance(selected, str) and selected in candidate_gradients and certificate_objects[selected].accepted_bottleneck:
            baseline_trial.restore()
            with trial_state(parameters, self.optimizer) as trial:
                apply_flat_gradient_step(parameters=parameters, flat_gradient=candidate_gradients[selected], step_size=step_size)
                trial.commit()
            return GuardrailDecision(
                handled_update=True,
                committed_update=selected,
                backtracks=0,
                fallback_used=True,
                skipped_update=False,
            )
        if certificate_objects and all(not certificate.accepted_bottleneck for certificate in certificate_objects.values()):
            baseline_trial.restore()
            return GuardrailDecision(
                handled_update=True,
                committed_update="skip_update",
                backtracks=0,
                fallback_used=False,
                skipped_update=True,
            )
        baseline_trial.restore()
        return default_guardrail_decision()

    def _certificate_for_current_state(
        self,
        *,
        probe_batches: list[ProbeBatch],
        context_ids: tuple[int, ...],
        old_context_losses: torch.Tensor,
        backtracks: int,
        step_size: float,
    ) -> StepCertificate:
        with torch.no_grad():
            losses = self._evaluate_context_losses(
                probe_batches=probe_batches,
                context_ids=context_ids,
            )
        return certify_losses(
            old_context_losses=old_context_losses,
            new_context_losses=losses,
            backtracks=backtracks,
            step_size=step_size,
            context_ids=context_ids,
        )

    def _evaluate_context_losses(
        self,
        *,
        probe_batches: list[ProbeBatch],
        context_ids: tuple[int, ...],
    ) -> torch.Tensor:
        loss_sums = {context_id: torch.tensor(0.0, dtype=torch.float32) for context_id in context_ids}
        counts = {context_id: 0 for context_id in context_ids}
        for probe_batch in probe_batches:
            device_timesteps = probe_batch.timesteps.to(device=self.unet.device).long()
            for context_id in context_ids:
                mask = device_timesteps == int(context_id)
                if not torch.any(mask):
                    continue
                example_count = int(mask.sum().item())
                loss = fixed_subset_context_loss(
                    batch=probe_batch.batch,
                    mask=mask,
                    unet=self.unet,
                    noise_scheduler=self.noise_scheduler,
                    weight_dtype=self.weight_dtype,
                    noise=probe_batch.noise,
                    timesteps=probe_batch.timesteps,
                )
                loss_sums[context_id] = loss_sums[context_id] + loss.detach().float().cpu() * float(example_count)
                counts[context_id] += example_count
        losses: list[torch.Tensor] = []
        for context_id in context_ids:
            if counts[context_id] <= 0:
                raise LorakitError(f"Probe context disappeared during candidate evaluation: {context_id}")
            losses.append(loss_sums[context_id] / float(counts[context_id]))
        return torch.stack(losses)


def candidate_gradients_from_probe(
    *,
    gradients: list[torch.Tensor],
    mgda_weights: torch.Tensor,
    context_counts: list[int],
) -> dict[str, torch.Tensor]:
    context_count = len(gradients)
    mean_context_weights = torch.full((context_count,), 1.0 / context_count, dtype=torch.float32)
    count_weights = torch.tensor(context_counts, dtype=torch.float32)
    count_weights = count_weights / count_weights.sum().clamp_min(1.0)
    return {
        "mean_context_sgd_proxy": weighted_gradient(gradients=gradients, weights=mean_context_weights),
        "mean_example_sgd_proxy": weighted_gradient(gradients=gradients, weights=count_weights),
        "mgda_sgd_proxy": weighted_gradient(gradients=gradients, weights=mgda_weights),
    }


def flatten_grads_from_autograd(
    *,
    parameters: list[torch.nn.Parameter],
    grads: tuple[torch.Tensor | None, ...],
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for parameter, grad in zip(parameters, grads, strict=True):
        if grad is None:
            pieces.append(torch.zeros(parameter.numel(), dtype=torch.float32))
        else:
            pieces.append(grad.detach().float().reshape(-1).cpu())
    if not pieces:
        raise LorakitError("No trainable parameters found for context probe")
    return torch.cat(pieces, dim=0)
