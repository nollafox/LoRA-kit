"""Validation-scored training policy challengers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import DDPMScheduler, UNet2DConditionModel

from lorakit.training.backends._evaluator import DiffusionValidator
from lorakit.training.backends._loss import loss_context_fixed
from lorakit.training.backends._optimizer import set_lora_plus_optimizer_ratio
from lorakit.training.backends._policy import (
    ObjectivePolicy,
    TrainingPolicy,
    lora_plus_candidate_ratios,
    lora_relative_gradient_scales,
    objective_candidates,
    ratio_label,
)
from lorakit.training.backends._trial import trial_state
from lorakit.training.backends._validation import (
    ValidationReport,
)
from lorakit.training.certified_stepper import trainable_parameters


@dataclass(frozen=True)
class FrozenBatch:
    batch: dict[str, object]
    noise: torch.Tensor
    timesteps: torch.Tensor


@dataclass
class PolicyChallenger:
    unet: UNet2DConditionModel
    optimizer: object
    validator: DiffusionValidator
    noise_scheduler: DDPMScheduler
    weight_dtype: torch.dtype
    learning_rate: float

    def choose(
        self,
        *,
        current_policy: TrainingPolicy,
        frozen_batch: FrozenBatch,
        baseline: ValidationReport,
    ) -> tuple[TrainingPolicy, dict[str, object]]:
        parameters = trainable_parameters(self.unet)
        baseline_trial = trial_state(parameters, self.optimizer)
        if not baseline_trial.has_optimizer_state:
            return current_policy, self._unavailable_report(current_policy)

        best_objective, objective_report = self._choose_objective(
            current_policy=current_policy,
            frozen_batch=frozen_batch,
            baseline=baseline,
            baseline_trial=baseline_trial,
        )
        best_ratio, ratio_report = self._choose_lora_plus_ratio(
            current_policy=current_policy,
            objective=best_objective,
            frozen_batch=frozen_batch,
            baseline=baseline,
            baseline_trial=baseline_trial,
            objective_baseline=objective_report.best_report,
        )

        baseline_trial.restore()
        set_lora_plus_optimizer_ratio(
            self.optimizer,
            base_learning_rate=self.learning_rate,
            ratio=best_ratio,
        )

        next_policy = TrainingPolicy(objective=best_objective, lora_plus_ratio=best_ratio)
        return next_policy, {
            "active_objective": current_policy.objective.label,
            "objective_candidates": objective_report.logs,
            "selected_objective": best_objective.label,
            "objective_switched": best_objective != current_policy.objective,
            "lora_plus": {
                "active_ratio": float(current_policy.lora_plus_ratio),
                "candidate_ratios": [float(value) for value in ratio_report.candidates],
                "selected_ratio": float(best_ratio),
                "ratio_switched": abs(float(best_ratio) - float(current_policy.lora_plus_ratio)) > 1e-12,
                "scale_A": float(ratio_report.scale_a),
                "scale_B": float(ratio_report.scale_b),
                "validation_scores": ratio_report.logs,
            },
        }

    def _choose_objective(
        self,
        *,
        current_policy: TrainingPolicy,
        frozen_batch: FrozenBatch,
        baseline: ValidationReport,
        baseline_trial,
    ) -> tuple[ObjectivePolicy, "_CandidateReport"]:
        logs: dict[str, object] = {}
        best_objective = current_policy.objective
        best_report = baseline
        for objective in objective_candidates(
            current_policy=current_policy,
            baseline_report=baseline,
            validation_skeleton=self.validator.skeleton,
            noise_scheduler=self.noise_scheduler,
        ):
            report = self._virtual_step_report(
                frozen_batch=frozen_batch,
                objective=objective,
                lora_plus_ratio=current_policy.lora_plus_ratio,
                baseline_trial=baseline_trial,
            )
            logs[objective.label] = report.delta_from(baseline, gamma=objective.gamma)
            if report.improves(best_report):
                best_objective = objective
                best_report = report
        return best_objective, _CandidateReport(logs=logs, best_report=best_report)

    def _choose_lora_plus_ratio(
        self,
        *,
        current_policy: TrainingPolicy,
        objective: ObjectivePolicy,
        frozen_batch: FrozenBatch,
        baseline: ValidationReport,
        baseline_trial,
        objective_baseline: ValidationReport,
    ) -> tuple[float, "_RatioReport"]:
        logs: dict[str, object] = {}
        best_ratio = current_policy.lora_plus_ratio
        best_report = objective_baseline
        scale_a, scale_b = lora_relative_gradient_scales(self.unet)
        candidates = lora_plus_candidate_ratios(self.unet)
        for ratio in candidates:
            report = self._virtual_step_report(
                frozen_batch=frozen_batch,
                objective=objective,
                lora_plus_ratio=ratio,
                baseline_trial=baseline_trial,
            )
            logs[ratio_label(ratio)] = report.delta_from(baseline)
            if report.improves(best_report):
                best_ratio = ratio
                best_report = report
        return best_ratio, _RatioReport(
            candidates=candidates,
            logs=logs,
            scale_a=scale_a,
            scale_b=scale_b,
        )

    def _virtual_step_report(
        self,
        *,
        frozen_batch: FrozenBatch,
        objective: ObjectivePolicy,
        lora_plus_ratio: float,
        baseline_trial,
    ) -> ValidationReport:
        parameters = trainable_parameters(self.unet)
        with trial_state(parameters, self.optimizer) as trial:
            baseline_trial.restore()
            set_lora_plus_optimizer_ratio(
                self.optimizer,
                base_learning_rate=self.learning_rate,
                ratio=lora_plus_ratio,
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss_context = loss_context_fixed(
                batch=frozen_batch.batch,
                unet=self.unet,
                noise_scheduler=self.noise_scheduler,
                weight_dtype=self.weight_dtype,
                objective=objective,
                noise=frozen_batch.noise,
                timesteps=frozen_batch.timesteps,
            )
            loss_context.loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                report = self.validator.score(step=-1)
            trial.restore()
            return report

    def _unavailable_report(self, current_policy: TrainingPolicy) -> dict[str, object]:
        return {
            "active_objective": current_policy.objective.label,
            "selected_objective": current_policy.objective.label,
            "objective_switched": False,
            "lora_plus": {
                "active_ratio": float(current_policy.lora_plus_ratio),
                "selected_ratio": float(current_policy.lora_plus_ratio),
                "ratio_switched": False,
                "reason": "optimizer_state_snapshot_unavailable",
            },
        }


@dataclass(frozen=True)
class _CandidateReport:
    logs: dict[str, object]
    best_report: ValidationReport


@dataclass(frozen=True)
class _RatioReport:
    candidates: list[float]
    logs: dict[str, object]
    scale_a: float
    scale_b: float
