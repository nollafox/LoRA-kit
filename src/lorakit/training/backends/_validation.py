"""Validation split, scoring, and reporting helpers for Diffusers training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

import torch
from diffusers import DDPMScheduler

from lorakit.errors import LorakitError
from lorakit.training.backends._cache import CacheRecord


VALIDATION_FRACTION: Final = 0.08
VALIDATION_MAX_ITEMS: Final = 32
VALIDATION_SCORE_ABSOLUTE_TOLERANCE: Final = 1e-7
VALIDATION_SCORE_RELATIVE_TOLERANCE: Final = 1e-5
VALIDATION_HIGH_SNR_FRACTION: Final = 0.10
VALIDATION_MID_SNR_FRACTION: Final = 0.50
VALIDATION_LOW_SNR_FRACTION: Final = 0.90
VALIDATION_SPLIT_NAMESPACE: Final = "lorakit-validation-v1"
VALIDATION_NOISE_NAMESPACE: Final = "lorakit-validation-noise-v1"
VALIDATION_SEED_HEX_LENGTH: Final = 16
VALIDATION_SEED_MODULUS: Final = 2**31


@dataclass(frozen=True)
class ValidationItem:
    record: CacheRecord
    timestep: int
    noise_seed: int
    snr: float
    snr_bucket: str


@dataclass(frozen=True)
class ValidationSkeleton:
    items: tuple[ValidationItem, ...]


@dataclass(frozen=True)
class ValidationReport:
    step: int
    loss_mean: float
    loss_max_snr_bucket: float
    loss_by_snr_bucket: dict[str, float]
    item_count: int

    @property
    def tolerance(self) -> float:
        return max(
            VALIDATION_SCORE_ABSOLUTE_TOLERANCE,
            VALIDATION_SCORE_RELATIVE_TOLERANCE * max(self.loss_by_snr_bucket.values()),
        )

    @property
    def worst_bucket(self) -> str:
        return max(self.loss_by_snr_bucket, key=self.loss_by_snr_bucket.get)

    def improves(self, baseline: "ValidationReport") -> bool:
        tolerance = baseline.tolerance
        if self.loss_max_snr_bucket < baseline.loss_max_snr_bucket - tolerance:
            return True
        if self.loss_max_snr_bucket > baseline.loss_max_snr_bucket + tolerance:
            return False
        return self.loss_mean < baseline.loss_mean - tolerance

    def nonworse_than(self, baseline: "ValidationReport") -> bool:
        tolerance = baseline.tolerance
        if self.loss_max_snr_bucket > baseline.loss_max_snr_bucket + tolerance:
            return False
        if self.loss_mean > baseline.loss_mean + tolerance:
            return False
        return True

    def delta_from(self, baseline: "ValidationReport", *, gamma: float | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "validation_loss_max_snr_bucket_delta": float(
                self.loss_max_snr_bucket - baseline.loss_max_snr_bucket
            ),
            "validation_loss_mean_delta": float(self.loss_mean - baseline.loss_mean),
        }
        if gamma is not None:
            payload["gamma"] = float(gamma)
        return payload

    def log_payload(self, *, best_checkpoint: "BestCheckpoint", improved: bool) -> dict[str, object]:
        return {
            "step": int(self.step),
            "probe_status": "validation",
            "validation_loss_mean": float(self.loss_mean),
            "validation_loss_max_snr_bucket": float(self.loss_max_snr_bucket),
            "validation_loss_by_snr_bucket": {
                bucket: float(loss)
                for bucket, loss in sorted(self.loss_by_snr_bucket.items())
            },
            "validation_item_count": int(self.item_count),
            "best_checkpoint_improved": bool(improved),
            "best_checkpoint_step": best_checkpoint.step,
            "best_checkpoint_path": str(best_checkpoint.path),
        }


@dataclass(frozen=True)
class BestCheckpoint:
    path: Path
    step: int | None
    loss_max_snr_bucket: float | None
    loss_mean: float | None

    @classmethod
    def empty(cls, path: Path) -> "BestCheckpoint":
        return cls(path=path, step=None, loss_max_snr_bucket=None, loss_mean=None)

    def consider(self, report: ValidationReport) -> tuple[bool, "BestCheckpoint"]:
        if self.loss_max_snr_bucket is None or self.loss_mean is None or self.step is None:
            return True, BestCheckpoint(
                path=self.path,
                step=report.step,
                loss_max_snr_bucket=report.loss_max_snr_bucket,
                loss_mean=report.loss_mean,
            )
        candidate_key = (report.loss_max_snr_bucket, report.loss_mean, report.step)
        current_key = (self.loss_max_snr_bucket, self.loss_mean, self.step)
        if candidate_key < current_key:
            return True, BestCheckpoint(
                path=self.path,
                step=report.step,
                loss_max_snr_bucket=report.loss_max_snr_bucket,
                loss_mean=report.loss_mean,
            )
        return False, self


def split_train_validation_records(
    records: list[CacheRecord],
) -> tuple[list[CacheRecord], list[CacheRecord]]:
    if len(records) < 2:
        return records, []
    validation_count = min(
        VALIDATION_MAX_ITEMS,
        max(1, int(round(len(records) * VALIDATION_FRACTION))),
    )
    validation_candidates = sorted(records, key=validation_split_key)[:validation_count]
    validation_hashes = {record.image_sha256 for record in validation_candidates}
    train_records = [record for record in records if record.image_sha256 not in validation_hashes]
    validation_records = [record for record in records if record.image_sha256 in validation_hashes]
    if not train_records:
        return records, []
    return train_records, validation_records


def validation_split_key(record: CacheRecord) -> str:
    return hashlib.sha256(
        f"{VALIDATION_SPLIT_NAMESPACE}:{record.image_sha256}".encode("utf-8")
    ).hexdigest()


def build_validation_skeleton(
    *,
    records: list[CacheRecord],
    noise_scheduler: DDPMScheduler,
) -> ValidationSkeleton:
    if not records:
        return ValidationSkeleton(items=())
    bucket_timesteps = validation_bucket_timesteps(
        total_timesteps=int(noise_scheduler.config.num_train_timesteps)
    )
    timesteps = [timestep for _record in records for timestep in bucket_timesteps.values()]
    snr_values = snr_for_timesteps(
        noise_scheduler=noise_scheduler,
        timesteps=torch.tensor(timesteps, dtype=torch.long),
    ).detach().float().cpu()
    bucket_names = [bucket for _record in records for bucket in bucket_timesteps]
    repeated_records = [record for record in records for _bucket in bucket_timesteps]
    items = tuple(
        ValidationItem(
            record=record,
            timestep=int(timestep),
            noise_seed=validation_noise_seed(record=record, snr_bucket=bucket),
            snr=float(snr),
            snr_bucket=bucket,
        )
        for record, timestep, bucket, snr in zip(
            repeated_records,
            timesteps,
            bucket_names,
            snr_values.tolist(),
            strict=True,
        )
    )
    return ValidationSkeleton(items=items)


def validation_bucket_timesteps(*, total_timesteps: int) -> dict[str, int]:
    if total_timesteps <= 0:
        raise LorakitError("Noise scheduler must expose at least one timestep")
    return {
        "high": validation_timestep_at_fraction(
            fraction=VALIDATION_HIGH_SNR_FRACTION,
            total_timesteps=total_timesteps,
        ),
        "mid": validation_timestep_at_fraction(
            fraction=VALIDATION_MID_SNR_FRACTION,
            total_timesteps=total_timesteps,
        ),
        "low": validation_timestep_at_fraction(
            fraction=VALIDATION_LOW_SNR_FRACTION,
            total_timesteps=total_timesteps,
        ),
    }


def validation_timestep_at_fraction(*, fraction: float, total_timesteps: int) -> int:
    if total_timesteps <= 0:
        raise LorakitError("Noise scheduler must expose at least one timestep")
    if fraction < 0.0 or fraction > 1.0:
        raise LorakitError(f"Validation timestep fraction must be within [0, 1]: {fraction}")
    return min(total_timesteps - 1, max(0, int(round(float(total_timesteps - 1) * fraction))))


def validation_noise_seed(*, record: CacheRecord, snr_bucket: str) -> int:
    digest = hashlib.sha256(
        f"{VALIDATION_NOISE_NAMESPACE}:{record.image_sha256}:{snr_bucket}".encode("utf-8")
    ).hexdigest()
    return int(digest[:VALIDATION_SEED_HEX_LENGTH], 16) % VALIDATION_SEED_MODULUS


def evaluate_validation_skeleton(
    *,
    skeleton: ValidationSkeleton,
    step: int,
    loss_for_item: Callable[[ValidationItem], float],
) -> ValidationReport:
    if not skeleton.items:
        raise LorakitError("Validation skeleton has no items")
    losses_by_bucket: dict[str, list[float]] = {}
    with torch.no_grad():
        for item in skeleton.items:
            loss = loss_for_item(item)
            losses_by_bucket.setdefault(item.snr_bucket, []).append(loss)
    bucket_means = {
        bucket: float(sum(losses) / len(losses))
        for bucket, losses in losses_by_bucket.items()
    }
    all_losses = [loss for losses in losses_by_bucket.values() for loss in losses]
    return ValidationReport(
        step=int(step),
        loss_mean=float(sum(all_losses) / len(all_losses)),
        loss_max_snr_bucket=float(max(bucket_means.values())),
        loss_by_snr_bucket=bucket_means,
        item_count=len(all_losses),
    )


def select_best_checkpoint(
    *,
    current: BestCheckpoint,
    report: ValidationReport,
) -> tuple[bool, BestCheckpoint]:
    return current.consider(report)


def write_validation_report_if_main(
    *,
    probe_log_path: Path,
    report: ValidationReport,
    best_checkpoint: BestCheckpoint,
    improved: bool,
    should_log: bool,
) -> None:
    if not should_log:
        return
    payload = report.log_payload(best_checkpoint=best_checkpoint, improved=improved)
    with probe_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def snr_for_timesteps(*, noise_scheduler: DDPMScheduler, timesteps: torch.Tensor) -> torch.Tensor:
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
    alpha = torch.sqrt(alphas_cumprod[timesteps])
    sigma = torch.sqrt(1.0 - alphas_cumprod[timesteps]).clamp_min(1e-12)
    return (alpha / sigma) ** 2
