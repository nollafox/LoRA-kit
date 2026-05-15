"""Diffusers LoRA training backend.

This backend is intentionally zero-config from the CLI side.  Internally it uses
heldout SNR-bucket validation to select checkpoints and to activate quality
challengers only when their virtual one-step validation score improves beyond a
scale-aware tolerance.

Key non-heuristic invariants:
- Latent cache is deterministic: posterior mode when available, otherwise mean.
- LoRA+ is implemented as actual AdamW param-group learning-rate ratios, not by
  scaling gradients before Adam's moment normalization.
- Objective / LoRA+ challenger comparisons use the same frozen training
  perturbation: identical batch, timesteps, and noise.
- Rank growth is triggered by validation plateau and uses dense base-weight
  validation gradients split into signal/noise estimates.  New rank channels are
  no-op initialized so model outputs are unchanged at insertion time.
"""

import copy
import gc
import json
import shutil
import random
from importlib.util import find_spec
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable

import torch
from accelerate import Accelerator
from bitsandbytes.optim import AdamW8bit
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers

from lorakit.errors import LorakitError
from lorakit.training.backends._cache import (
    CACHE_DIR_NAME,
    CACHE_MANIFEST_NAME,
    CacheRecord as _CacheRecord,
    DiskCachedLatentDataset as _DiskCachedLatentDataset,
    LatentShapeBatchSampler as _LatentShapeBatchSampler,
    cache_encoder_hidden_states as _cache_encoder_hidden_states,
    cache_latents as _cache_latents,
    cache_records as _cache_records,
    cleanup_after_cuda_oom as _cleanup_after_cuda_oom,
    collate_cached as _collate_cached,
    is_cuda_oom as _is_cuda_oom,
    load_rows as _load_rows,
    load_tensor as _load_tensor,
    write_cache_manifest as _write_cache_manifest,
)
from lorakit.training.backends._validation import (
    BestCheckpoint as _BestCheckpoint,
    ValidationItem as _ValidationItem,
    ValidationReport as _ValidationReport,
    ValidationSkeleton as _ValidationSkeleton,
    build_validation_skeleton as _build_validation_skeleton,
    evaluate_validation_skeleton as _evaluate_validation_skeleton_from_items,
    select_best_checkpoint as _select_best_checkpoint,
    snr_for_timesteps as _snr_for_timesteps,
    split_train_validation_records as _split_train_validation_records,
    validation_delta_log as _validation_delta_log,
    validation_score_improves as _validation_score_improves,
    validation_score_nonworse as _validation_score_nonworse,
    write_validation_report_if_main as _write_validation_report_if_main,
)
from lorakit.training.backends.types import BackendResult, BackendSpec
from lorakit.training.certified_stepper import (
    StepCertificate,
    candidate_first_order_summaries,
    certificate_to_log_dict,
    certify_losses,
    denoising_loss_per_example,
    probe_from_context_losses,
    probe_to_log_dict,
    select_preferred_candidate,
    trainable_parameters as certified_trainable_parameters,
    weighted_gradient,
)


LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]
LORA_WEIGHTS_NAME = "pytorch_lora_weights.safetensors"
BEST_LORA_WEIGHTS_NAME = "best_lora_weights.safetensors"
MAX_GRAD_NORM = 1.0
LR_WARMUP_STEPS = 0

VALIDATION_EVERY_PROBES: Final = 1

RANK_GROWTH_MAX_CHANNELS: Final = 2
RANK_GROWTH_MIN_FREE_CUDA_BYTES: Final = 1_000_000_000
RANK_GROWTH_NOISE_MULTIPLIER: Final = 2.858


PROBE_LOG_NAME = "context-probes.jsonl"
PROBE_SUMMARY_NAME = "context-probes-summary.json"
PROBE_INITIAL_STEPS: Final = 3
PROBE_EVERY_STEPS: Final = 25
PROBE_MIN_FREE_CUDA_BYTES: Final = 500_000_000
PROBE_WINDOW_TARGET_ITEMS: Final = 4
PROBE_VERSION: Final = 5
CERTIFIED_BACKTRACK_FACTORS: Final = (1.0, 0.5, 0.25, 0.125)


@dataclass(frozen=True)
class _ObjectivePolicy:
    name: str
    gamma: float | None = None


@dataclass(frozen=True)
class _TrainingPolicy:
    objective: _ObjectivePolicy
    lora_plus_ratio: float


@dataclass(frozen=True)
class _LossContext:
    loss: torch.Tensor
    noise: torch.Tensor
    timesteps: torch.Tensor


@dataclass(frozen=True)
class _ProbeBatch:
    batch: dict[str, object]
    noise: torch.Tensor
    timesteps: torch.Tensor
    source: str


@dataclass(frozen=True)
class _GuardrailDecision:
    handled_update: bool
    committed_update: str
    backtracks: int
    fallback_used: bool
    skipped_update: bool


@dataclass(frozen=True)
class _RankGrowthProposal:
    report: dict[str, object]
    init_a_rows: torch.Tensor | None


class _Components:
    def __init__(
        self,
        *,
        noise_scheduler: DDPMScheduler,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
    ):
        self.noise_scheduler = noise_scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet


def _default_guardrail_decision() -> _GuardrailDecision:
    return _GuardrailDecision(
        handled_update=False,
        committed_update="adamw_actual",
        backtracks=0,
        fallback_used=False,
        skipped_update=False,
    )


def _default_training_policy() -> _TrainingPolicy:
    return _TrainingPolicy(objective=_ObjectivePolicy(name="base_mse"), lora_plus_ratio=1.0)


def train(spec: BackendSpec) -> BackendResult:
    _validate_spec(spec)
    rows = _load_rows(spec.prepared_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=spec.gradient_accumulation,
        mixed_precision=_accelerate_mixed_precision(spec.mixed_precision),
        project_dir=spec.output_dir,
    )
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    components = _load_components(spec.model)
    noise_scheduler = components.noise_scheduler
    tokenizer = components.tokenizer
    text_encoder = components.text_encoder
    vae = components.vae
    unet = components.unet

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.add_adapter(
        LoraConfig(
            r=spec.rank,
            lora_alpha=spec.rank,
            init_lora_weights="gaussian",
            target_modules=LORA_TARGET_MODULES,
        )
    )
    unet.enable_gradient_checkpointing()
    attention_backend = _enable_memory_efficient_attention(unet)

    weight_dtype = _weight_dtype(accelerator)
    cache_dir = spec.output_dir / CACHE_DIR_NAME

    encoder_hidden_state_paths = _cache_encoder_hidden_states(
        rows=rows,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=accelerator.device,
        dtype=weight_dtype,
    )
    latent_paths = _cache_latents(
        rows=rows,
        cache_dir=cache_dir,
        vae=vae,
        device=accelerator.device,
        dtype=weight_dtype,
    )
    latent_scaling_factor = float(vae.config.scaling_factor)

    text_encoder.to("cpu")
    vae.to("cpu")
    del text_encoder
    del vae
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    unet.to(accelerator.device, dtype=weight_dtype)
    if accelerator.mixed_precision == "fp16":
        _cast_trainable_parameters_to_fp32(unet)
        _assert_trainable_parameters_are_fp32(unet)

    cache_records = _cache_records(
        rows=rows,
        latent_paths=latent_paths,
        encoder_hidden_state_paths=encoder_hidden_state_paths,
    )
    _write_cache_manifest(
        cache_dir / CACHE_MANIFEST_NAME,
        cache_records,
        latent_scaling_factor=latent_scaling_factor,
    )

    train_records, validation_records = _split_train_validation_records(cache_records)
    validation_skeleton = _build_validation_skeleton(
        records=validation_records,
        noise_scheduler=noise_scheduler,
    )
    best_checkpoint = _BestCheckpoint.empty(spec.output_dir / BEST_LORA_WEIGHTS_NAME)
    validation_probe_count = 0
    training_policy = _default_training_policy()

    dataset = _DiskCachedLatentDataset(records=train_records)
    dataloader = DataLoader(
        dataset,
        batch_sampler=_LatentShapeBatchSampler(records=train_records, batch_size=spec.batch_size),
        collate_fn=_collate_cached,
        num_workers=0,
        pin_memory=False,
    )

    optimizer = _optimizer(unet, spec.learning_rate)
    scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=LR_WARMUP_STEPS,
        num_training_steps=spec.steps * accelerator.num_processes,
    )
    unet, optimizer, dataloader, scheduler = accelerator.prepare(
        unet, optimizer, dataloader, scheduler
    )

    global_step = 0
    if accelerator.is_local_main_process:
        print(f"Attention backend: {attention_backend}")

    progress = tqdm(
        range(spec.steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    probe_log_path = spec.output_dir / PROBE_LOG_NAME
    probe_summary_path = spec.output_dir / PROBE_SUMMARY_NAME

    while global_step < spec.steps:
        for batch in dataloader:
            guardrail_decision = _default_guardrail_decision()
            with accelerator.accumulate(unet):
                _set_lora_plus_optimizer_ratio(
                    optimizer,
                    base_learning_rate=spec.learning_rate,
                    ratio=training_policy.lora_plus_ratio,
                )
                loss_context = _loss_context(
                    batch=batch,
                    unet=unet,
                    noise_scheduler=noise_scheduler,
                    weight_dtype=weight_dtype,
                    objective=training_policy.objective,
                )
                loss = loss_context.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in unet.parameters() if p.requires_grad],
                        MAX_GRAD_NORM,
                    )

                should_probe = _should_probe_contexts(
                    global_step=global_step,
                    sync_gradients=accelerator.sync_gradients,
                )

                if should_probe:
                    cuda_memory_before_probe = _cuda_memory_snapshot()
                    _write_probe_status_if_main(
                        probe_log_path=probe_log_path,
                        step=global_step,
                        should_log=accelerator.is_local_main_process,
                        status="started",
                        free_cuda_bytes=(
                            cuda_memory_before_probe["free_cuda_bytes"]
                            if cuda_memory_before_probe is not None
                            else None
                        ),
                        extra={
                            "probe_version": PROBE_VERSION,
                            "microbatch_size": int(loss_context.noise.shape[0]),
                            "requested_batch_size": int(spec.batch_size),
                            "gradient_accumulation": int(spec.gradient_accumulation),
                            "active_objective": _objective_label(training_policy.objective),
                            "active_lora_plus_ratio": float(training_policy.lora_plus_ratio),
                        },
                    )
                    try:
                        guardrail_decision = _write_streaming_context_probe_if_main(
                            unet=unet,
                            optimizer=optimizer,
                            dataset=dataset,
                            batch=batch,
                            noise_scheduler=noise_scheduler,
                            weight_dtype=weight_dtype,
                            noise=loss_context.noise,
                            timesteps=loss_context.timesteps,
                            probe_log_path=probe_log_path,
                            step=global_step,
                            should_log=accelerator.is_local_main_process,
                            requested_batch_size=spec.batch_size,
                            gradient_accumulation=spec.gradient_accumulation,
                            training_loss=loss,
                            candidate_step_size=spec.learning_rate,
                            cuda_memory_before_probe=cuda_memory_before_probe,
                        )
                    except RuntimeError as exc:
                        if not _is_cuda_oom(exc):
                            raise
                        _cleanup_after_cuda_oom()
                        _write_probe_status_if_main(
                            probe_log_path=probe_log_path,
                            step=global_step,
                            should_log=accelerator.is_local_main_process,
                            status="skipped_cuda_oom",
                            free_cuda_bytes=_cuda_free_bytes(),
                            extra={
                                "probe_version": PROBE_VERSION,
                                "requested_batch_size": int(spec.batch_size),
                                "gradient_accumulation": int(spec.gradient_accumulation),
                            },
                        )
                elif accelerator.sync_gradients and accelerator.is_local_main_process:
                    free_cuda_bytes = _cuda_free_bytes()
                    if free_cuda_bytes is not None and free_cuda_bytes < PROBE_MIN_FREE_CUDA_BYTES:
                        _write_probe_status_if_main(
                            probe_log_path=probe_log_path,
                            step=global_step,
                            should_log=True,
                            status="skipped_low_cuda_memory",
                            free_cuda_bytes=free_cuda_bytes,
                            extra={
                                "probe_version": PROBE_VERSION,
                                "requested_batch_size": int(spec.batch_size),
                                "gradient_accumulation": int(spec.gradient_accumulation),
                            },
                        )

                committed_update = not guardrail_decision.skipped_update
                if not guardrail_decision.handled_update:
                    _set_lora_plus_optimizer_ratio(
                        optimizer,
                        base_learning_rate=spec.learning_rate,
                        ratio=training_policy.lora_plus_ratio,
                    )
                    optimizer.step()

                if committed_update:
                    scheduler.step()

                if should_probe and committed_update and validation_skeleton.items:
                    validation_probe_count += 1
                    if validation_probe_count % VALIDATION_EVERY_PROBES == 0:
                        validation_report = _evaluate_validation_skeleton(
                            skeleton=validation_skeleton,
                            unet=unet,
                            noise_scheduler=noise_scheduler,
                            weight_dtype=weight_dtype,
                            step=global_step,
                        )
                        improved, best_checkpoint = _select_best_checkpoint(
                            current=best_checkpoint,
                            report=validation_report,
                        )
                        if improved and accelerator.is_main_process:
                            _save_named_lora_weights(
                                unet=accelerator.unwrap_model(unet),
                                output_dir=spec.output_dir,
                                filename=BEST_LORA_WEIGHTS_NAME,
                            )
                        _write_validation_report_if_main(
                            probe_log_path=probe_log_path,
                            report=validation_report,
                            best_checkpoint=best_checkpoint,
                            improved=improved,
                            should_log=accelerator.is_local_main_process,
                        )

                        training_policy, challenger_log = _challenge_quality_policies(
                            current_policy=training_policy,
                            batch=batch,
                            candidate_noise=loss_context.noise,
                            candidate_timesteps=loss_context.timesteps,
                            unet=unet,
                            optimizer=optimizer,
                            validation_skeleton=validation_skeleton,
                            baseline_report=validation_report,
                            noise_scheduler=noise_scheduler,
                            weight_dtype=weight_dtype,
                            learning_rate=spec.learning_rate,
                        )
                        challenger_log["rank_probe"] = _maybe_grow_lora_rank(
                            unet=unet,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            validation_skeleton=validation_skeleton,
                            baseline_report=validation_report,
                            noise_scheduler=noise_scheduler,
                            weight_dtype=weight_dtype,
                            should_probe=not improved,
                            base_learning_rate=spec.learning_rate,
                            lora_plus_ratio=training_policy.lora_plus_ratio,
                        )
                        _write_policy_challenger_report_if_main(
                            probe_log_path=probe_log_path,
                            step=global_step,
                            payload=challenger_log,
                            should_log=accelerator.is_local_main_process,
                        )

                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=float(loss.detach().item()))
                if global_step >= spec.steps:
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        best_model_path = spec.output_dir / BEST_LORA_WEIGHTS_NAME
        if best_model_path.exists():
            shutil.copy2(best_model_path, spec.output_dir / LORA_WEIGHTS_NAME)
        else:
            _save_named_lora_weights(
                unet=accelerator.unwrap_model(unet),
                output_dir=spec.output_dir,
                filename=LORA_WEIGHTS_NAME,
            )
    accelerator.end_training()

    model_path = spec.output_dir / LORA_WEIGHTS_NAME
    if not model_path.exists():
        raise LorakitError(f"Diffusers backend did not write LoRA weights: {model_path}")
    if probe_log_path.exists():
        _write_probe_summary(probe_log_path=probe_log_path, summary_path=probe_summary_path)
    artifact_paths = tuple(
        path
        for path in (probe_log_path, probe_summary_path, spec.output_dir / BEST_LORA_WEIGHTS_NAME)
        if path.exists()
    )
    return BackendResult(model_path=model_path, artifact_paths=artifact_paths)


# ---------------------------------------------------------------------------
# Validation and loading
# ---------------------------------------------------------------------------



def _validate_spec(spec: BackendSpec) -> None:
    if spec.steps <= 0:
        raise LorakitError("Training steps must be greater than zero")
    if spec.rank <= 0:
        raise LorakitError("LoRA rank must be greater than zero")
    if spec.batch_size <= 0:
        raise LorakitError("Batch size must be greater than zero")
    if spec.gradient_accumulation <= 0:
        raise LorakitError("Gradient accumulation must be greater than zero")
    if spec.resolution <= 0:
        raise LorakitError("Resolution must be greater than zero")


def _enable_memory_efficient_attention(unet: UNet2DConditionModel) -> str:
    if find_spec("xformers") is None:
        return "torch"
    unet.enable_xformers_memory_efficient_attention()
    return "xformers"


def _cast_trainable_parameters_to_fp32(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)


def _assert_trainable_parameters_are_fp32(module: torch.nn.Module) -> None:
    bad = [
        (name, parameter.dtype)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.dtype != torch.float32
    ]
    if bad:
        preview = ", ".join(f"{name}: {dtype}" for name, dtype in bad[:5])
        raise LorakitError(
            "Trainable parameters must be fp32 under fp16 mixed precision. "
            f"Found: {preview}"
        )


def _load_single_file_components(model_path: Path) -> _Components:
    pipeline = StableDiffusionPipeline.from_single_file(
        str(model_path),
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    return _Components(
        noise_scheduler=DDPMScheduler.from_config(pipeline.scheduler.config),
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        vae=pipeline.vae,
        unet=pipeline.unet,
    )


def _load_components(model: str) -> _Components:
    model_path = Path(model)
    if model_path.exists() and model_path.is_file():
        return _load_single_file_components(model_path)

    model_ref = str(model_path) if model_path.exists() else model
    return _Components(
        noise_scheduler=DDPMScheduler.from_pretrained(model_ref, subfolder="scheduler"),
        tokenizer=CLIPTokenizer.from_pretrained(model_ref, subfolder="tokenizer"),
        text_encoder=CLIPTextModel.from_pretrained(
            model_ref,
            subfolder="text_encoder",
            low_cpu_mem_usage=False,
        ),
        vae=AutoencoderKL.from_pretrained(
            model_ref,
            subfolder="vae",
            low_cpu_mem_usage=False,
        ),
        unet=UNet2DConditionModel.from_pretrained(
            model_ref,
            subfolder="unet",
            low_cpu_mem_usage=False,
        ),
    )


def _accelerate_mixed_precision(value: str) -> str:
    if value == "no":
        return "no"
    if value in {"fp16", "bf16"}:
        return value
    raise LorakitError(f"Unsupported mixed precision: {value}")


def _weight_dtype(accelerator: Accelerator) -> torch.dtype:
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Objective, optimizer, and LoRA+
# ---------------------------------------------------------------------------


def _optimizer(parameters_or_unet, learning_rate: float):
    if isinstance(parameters_or_unet, torch.nn.Module):
        groups = _lora_plus_optimizer_groups(
            parameters_or_unet,
            base_learning_rate=learning_rate,
            ratio=1.0,
        )
        if not groups:
            groups = [
                {
                    "params": [
                        parameter
                        for parameter in parameters_or_unet.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": learning_rate,
                    "lorakit_group": "default",
                }
            ]
    else:
        groups = [
            {
                "params": list(parameters_or_unet),
                "lr": learning_rate,
                "lorakit_group": "default",
            }
        ]

    return AdamW8bit(
        groups,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )


def _lora_plus_optimizer_groups(
    unet: torch.nn.Module,
    *,
    base_learning_rate: float,
    ratio: float,
) -> list[dict[str, object]]:
    lora_a: list[torch.nn.Parameter] = []
    lora_b: list[torch.nn.Parameter] = []
    other: list[torch.nn.Parameter] = []
    for name, parameter in unet.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_A" in name:
            lora_a.append(parameter)
        elif "lora_B" in name:
            lora_b.append(parameter)
        else:
            other.append(parameter)

    groups: list[dict[str, object]] = []
    if lora_a:
        groups.append({"params": lora_a, "lr": float(base_learning_rate), "lorakit_group": "lora_A"})
    if lora_b:
        groups.append({"params": lora_b, "lr": float(base_learning_rate) * float(ratio), "lorakit_group": "lora_B"})
    if other:
        groups.append({"params": other, "lr": float(base_learning_rate), "lorakit_group": "default"})
    return groups


def _set_lora_plus_optimizer_ratio(optimizer, *, base_learning_rate: float, ratio: float) -> None:
    for group in optimizer.param_groups:
        tag = group.get("lorakit_group")
        if tag == "lora_B":
            group["lr"] = float(base_learning_rate) * float(ratio)
        elif tag in {"lora_A", "default", None}:
            group["lr"] = float(base_learning_rate)


def _add_lora_named_parameters_to_optimizer(
    optimizer,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    *,
    base_learning_rate: float,
    ratio: float,
) -> None:
    a_params = [p for name, p in named_parameters if "lora_A" in name and p.requires_grad]
    b_params = [p for name, p in named_parameters if "lora_B" in name and p.requires_grad]
    other_params = [
        parameter
        for name, parameter in named_parameters
        if "lora_A" not in name
        and "lora_B" not in name
        and parameter.requires_grad
    ]
    if a_params:
        optimizer.add_param_group(
            {
                "params": a_params,
                "lr": float(base_learning_rate),
                "initial_lr": float(base_learning_rate),
                "lorakit_group": "lora_A",
            }
        )
    if b_params:
        optimizer.add_param_group(
            {
                "params": b_params,
                "lr": float(base_learning_rate) * float(ratio),
                "initial_lr": float(base_learning_rate) * float(ratio),
                "lorakit_group": "lora_B",
            }
        )
    if other_params:
        optimizer.add_param_group(
            {
                "params": other_params,
                "lr": float(base_learning_rate),
                "initial_lr": float(base_learning_rate),
                "lorakit_group": "default",
            }
        )


def _sync_scheduler_param_groups_after_optimizer_growth(
    *,
    scheduler,
    optimizer,
    groups_before: int,
) -> None:
    """Keep torch/accelerate schedulers consistent after dynamic optimizer growth.

    PyTorch LR schedulers keep one base LR and, for LambdaLR, one lambda function
    per optimizer parameter group. Dynamic rank growth adds optimizer groups after
    scheduler creation, so the scheduler must receive matching entries before its
    next step().

    Raises:
        LorakitError: If optimizer or scheduler state is internally inconsistent.
    """
    groups_after = len(optimizer.param_groups)
    if groups_after <= groups_before:
        return

    wrapped_scheduler = getattr(scheduler, "scheduler", scheduler)
    _validate_scheduler_group_count(
        scheduler=wrapped_scheduler,
        optimizer=optimizer,
        groups_before=groups_before,
    )

    new_groups = optimizer.param_groups[groups_before:]
    new_base_lrs = [_learning_rate_for_scheduler_group(group) for group in new_groups]

    if hasattr(wrapped_scheduler, "base_lrs"):
        wrapped_scheduler.base_lrs.extend(new_base_lrs)

    if hasattr(wrapped_scheduler, "lr_lambdas"):
        _extend_scheduler_lambdas(
            scheduler=wrapped_scheduler,
            groups_before=groups_before,
            groups_after=groups_after,
        )

    if hasattr(wrapped_scheduler, "_last_lr"):
        last_lr = list(getattr(wrapped_scheduler, "_last_lr"))
        last_lr.extend(_current_learning_rate_for_scheduler_group(group) for group in new_groups)
        wrapped_scheduler._last_lr = last_lr


def _validate_scheduler_group_count(
    *,
    scheduler,
    optimizer,
    groups_before: int,
) -> None:
    base_lrs = getattr(scheduler, "base_lrs", None)
    if base_lrs is not None and len(base_lrs) != groups_before:
        raise LorakitError(
            "Scheduler base learning-rate count does not match optimizer groups "
            f"before rank growth: scheduler={len(base_lrs)}, optimizer_before={groups_before}, "
            f"optimizer_after={len(optimizer.param_groups)}"
        )

    lr_lambdas = getattr(scheduler, "lr_lambdas", None)
    if lr_lambdas is not None and len(lr_lambdas) != groups_before:
        raise LorakitError(
            "Scheduler lambda count does not match optimizer groups before rank growth: "
            f"scheduler={len(lr_lambdas)}, optimizer_before={groups_before}, "
            f"optimizer_after={len(optimizer.param_groups)}"
        )


def _learning_rate_for_scheduler_group(group: dict[str, object]) -> float:
    if "initial_lr" not in group:
        raise LorakitError("New optimizer parameter group is missing required initial_lr")
    return _float_optimizer_group_value(group=group, key="initial_lr")


def _current_learning_rate_for_scheduler_group(group: dict[str, object]) -> float:
    if "lr" not in group:
        raise LorakitError("New optimizer parameter group is missing required lr")
    return _float_optimizer_group_value(group=group, key="lr")


def _float_optimizer_group_value(*, group: dict[str, object], key: str) -> float:
    value = group[key]
    if not isinstance(value, int | float):
        raise LorakitError(f"Optimizer parameter group {key} must be numeric, got {type(value).__name__}")
    return float(value)


def _extend_scheduler_lambdas(
    *,
    scheduler,
    groups_before: int,
    groups_after: int,
) -> None:
    if groups_before == 0:
        raise LorakitError("Cannot extend scheduler lambdas without existing optimizer groups")

    lr_lambdas = list(scheduler.lr_lambdas)
    lambda_template = lr_lambdas[-1]
    lr_lambdas.extend(lambda_template for _ in range(groups_after - groups_before))
    scheduler.lr_lambdas = lr_lambdas


def _batch_tensor(batch: dict[str, object], key: str) -> torch.Tensor:
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise LorakitError(f"Batch {key} must be a tensor")
    return value


def _loss_context(
    *,
    batch: dict[str, object],
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    objective: _ObjectivePolicy,
) -> _LossContext:
    latents = _batch_tensor(batch, "latents").to(device=unet.device, dtype=weight_dtype)
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=latents.device,
    ).long()
    return _loss_context_fixed(
        batch=batch,
        unet=unet,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
        objective=objective,
        noise=noise,
        timesteps=timesteps,
    )


def _loss_context_fixed(
    *,
    batch: dict[str, object],
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    objective: _ObjectivePolicy,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> _LossContext:
    latents = _batch_tensor(batch, "latents").to(device=unet.device, dtype=weight_dtype)
    encoder_hidden_states = _batch_tensor(batch, "encoder_hidden_states").to(device=unet.device, dtype=weight_dtype)
    local_noise = noise.to(device=unet.device, dtype=weight_dtype)
    local_timesteps = timesteps.to(device=unet.device).long()
    noisy_latents = noise_scheduler.add_noise(latents, local_noise, local_timesteps)
    target = _target(
        noise_scheduler=noise_scheduler,
        latents=latents,
        noise=local_noise,
        timesteps=local_timesteps,
    )
    prediction = unet(noisy_latents, local_timesteps, encoder_hidden_states, return_dict=False)[0]
    per_example_loss = denoising_loss_per_example(model_pred=prediction, target=target)
    loss = _objective_loss(
        per_example_loss=per_example_loss,
        timesteps=local_timesteps,
        noise_scheduler=noise_scheduler,
        prediction_type=str(noise_scheduler.config.prediction_type),
        objective=objective,
    )
    return _LossContext(
        loss=loss,
        noise=local_noise.detach().cpu(),
        timesteps=local_timesteps.detach().cpu(),
    )


def _objective_loss(
    *,
    per_example_loss: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: DDPMScheduler,
    prediction_type: str,
    objective: _ObjectivePolicy,
) -> torch.Tensor:
    if objective.name == "base_mse":
        return per_example_loss.mean()
    if objective.name == "minsnr":
        if objective.gamma is None:
            raise LorakitError("Min-SNR objective requires gamma")
        weights = _snr_loss_weights(
            timesteps=timesteps,
            noise_scheduler=noise_scheduler,
            gamma=objective.gamma,
            prediction_type=prediction_type,
        )
        return torch.mean(per_example_loss * weights.to(device=per_example_loss.device, dtype=per_example_loss.dtype))
    raise LorakitError(f"Unknown training objective: {objective.name}")


def _snr_loss_weights(
    *,
    timesteps: torch.Tensor,
    noise_scheduler: DDPMScheduler,
    gamma: float,
    prediction_type: str,
) -> torch.Tensor:
    if gamma <= 0.0:
        raise LorakitError(f"Min-SNR gamma must be positive: {gamma}")
    snr = _snr_for_timesteps(
        noise_scheduler=noise_scheduler,
        timesteps=timesteps.detach().long().cpu(),
    ).to(device=timesteps.device, dtype=torch.float32)
    clipped = torch.minimum(snr, torch.tensor(float(gamma), device=snr.device, dtype=snr.dtype))
    if prediction_type == "epsilon":
        return clipped / snr.clamp_min(1e-12)
    if prediction_type == "v_prediction":
        return clipped / (snr + 1.0).clamp_min(1e-12)
    raise LorakitError(f"Unknown prediction type: {prediction_type}")


def _target(
    noise_scheduler: DDPMScheduler,
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    prediction_type = noise_scheduler.config.prediction_type
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        return noise_scheduler.get_velocity(latents, noise, timesteps)
    raise LorakitError(f"Unknown scheduler prediction type: {prediction_type}")


# ---------------------------------------------------------------------------
# Validation skeleton and checkpoint selection
# ---------------------------------------------------------------------------


def _evaluate_validation_skeleton(
    *,
    skeleton: _ValidationSkeleton,
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    step: int,
) -> _ValidationReport:
    return _evaluate_validation_skeleton_from_items(
        skeleton=skeleton,
        step=step,
        loss_for_item=lambda item: _validation_item_loss(
            item=item,
            unet=unet,
            noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
        ),
    )


@torch.no_grad()
def _validation_item_loss(
    *,
    item: _ValidationItem,
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
) -> float:
    loss = _validation_item_loss_tensor(
        item=item,
        unet=unet,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
    )
    return float(loss.detach().float().cpu().item())


def _validation_item_loss_tensor(
    *,
    item: _ValidationItem,
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    latent = _load_tensor(item.record.latent_path, expected_shape=item.record.latent_shape)
    hidden = _load_tensor(
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
    return _fixed_subset_context_loss(
        batch=batch,
        mask=torch.tensor([True], device=unet.device),
        unet=unet,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
        noise=noise.unsqueeze(0),
        timesteps=torch.tensor([item.timestep], dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Quality challengers
# ---------------------------------------------------------------------------


def _challenge_quality_policies(
    *,
    current_policy: _TrainingPolicy,
    batch: dict[str, object],
    candidate_noise: torch.Tensor,
    candidate_timesteps: torch.Tensor,
    unet: UNet2DConditionModel,
    optimizer,
    validation_skeleton: _ValidationSkeleton,
    baseline_report: _ValidationReport,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    learning_rate: float,
) -> tuple[_TrainingPolicy, dict[str, object]]:
    parameters = certified_trainable_parameters(unet)
    parameter_snapshot = _snapshot_trainable_parameters(parameters)
    gradient_snapshot = _snapshot_trainable_gradients(parameters)
    optimizer_snapshot = _snapshot_optimizer_state(optimizer)
    if optimizer_snapshot is None:
        return current_policy, {
            "active_objective": _objective_label(current_policy.objective),
            "selected_objective": _objective_label(current_policy.objective),
            "objective_switched": False,
            "lora_plus": {
                "active_ratio": float(current_policy.lora_plus_ratio),
                "selected_ratio": float(current_policy.lora_plus_ratio),
                "ratio_switched": False,
                "reason": "optimizer_state_snapshot_unavailable",
            },
        }

    objective_candidates = _objective_candidates(
        current_policy=current_policy,
        baseline_report=baseline_report,
        validation_skeleton=validation_skeleton,
        noise_scheduler=noise_scheduler,
    )
    objective_reports: dict[str, object] = {}
    best_objective = current_policy.objective
    best_objective_report = baseline_report
    for objective in objective_candidates:
        report = _virtual_policy_step_validation_report(
            batch=batch,
            candidate_noise=candidate_noise,
            candidate_timesteps=candidate_timesteps,
            unet=unet,
            optimizer=optimizer,
            parameters=parameters,
            parameter_snapshot=parameter_snapshot,
            gradient_snapshot=gradient_snapshot,
            optimizer_snapshot=optimizer_snapshot,
            validation_skeleton=validation_skeleton,
            noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
            objective=objective,
            lora_plus_ratio=current_policy.lora_plus_ratio,
            learning_rate=learning_rate,
        )
        objective_reports[_objective_label(objective)] = _validation_delta_log(
            baseline=baseline_report,
            candidate=report,
            gamma=objective.gamma,
        )
        if _validation_score_improves(baseline=best_objective_report, candidate=report):
            best_objective = objective
            best_objective_report = report

    ratio_candidates = _lora_plus_candidate_ratios(unet)
    ratio_reports: dict[str, object] = {}
    best_ratio = current_policy.lora_plus_ratio
    best_ratio_report = best_objective_report
    scale_a, scale_b = _lora_relative_gradient_scales(unet)
    for ratio in ratio_candidates:
        report = _virtual_policy_step_validation_report(
            batch=batch,
            candidate_noise=candidate_noise,
            candidate_timesteps=candidate_timesteps,
            unet=unet,
            optimizer=optimizer,
            parameters=parameters,
            parameter_snapshot=parameter_snapshot,
            gradient_snapshot=gradient_snapshot,
            optimizer_snapshot=optimizer_snapshot,
            validation_skeleton=validation_skeleton,
            noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
            objective=best_objective,
            lora_plus_ratio=ratio,
            learning_rate=learning_rate,
        )
        ratio_reports[_ratio_label(ratio)] = _validation_delta_log(
            baseline=baseline_report,
            candidate=report,
        )
        if _validation_score_improves(baseline=best_ratio_report, candidate=report):
            best_ratio = ratio
            best_ratio_report = report

    _restore_trainable_parameters(parameters, parameter_snapshot)
    _restore_trainable_gradients(parameters, gradient_snapshot)
    _restore_optimizer_state(optimizer, optimizer_snapshot)
    _set_lora_plus_optimizer_ratio(optimizer, base_learning_rate=learning_rate, ratio=best_ratio)

    next_policy = _TrainingPolicy(objective=best_objective, lora_plus_ratio=best_ratio)
    return next_policy, {
        "active_objective": _objective_label(current_policy.objective),
        "objective_candidates": objective_reports,
        "selected_objective": _objective_label(best_objective),
        "objective_switched": _objective_label(best_objective) != _objective_label(current_policy.objective),
        "lora_plus": {
            "active_ratio": float(current_policy.lora_plus_ratio),
            "candidate_ratios": [float(value) for value in ratio_candidates],
            "selected_ratio": float(best_ratio),
            "ratio_switched": abs(float(best_ratio) - float(current_policy.lora_plus_ratio)) > 1e-12,
            "scale_A": float(scale_a),
            "scale_B": float(scale_b),
            "validation_scores": ratio_reports,
        },
    }


def _virtual_policy_step_validation_report(
    *,
    batch: dict[str, object],
    candidate_noise: torch.Tensor,
    candidate_timesteps: torch.Tensor,
    unet: UNet2DConditionModel,
    optimizer,
    parameters: list[torch.nn.Parameter],
    parameter_snapshot: list[torch.Tensor],
    gradient_snapshot: list[torch.Tensor | None],
    optimizer_snapshot,
    validation_skeleton: _ValidationSkeleton,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    objective: _ObjectivePolicy,
    lora_plus_ratio: float,
    learning_rate: float,
) -> _ValidationReport:
    _restore_trainable_parameters(parameters, parameter_snapshot)
    _restore_trainable_gradients(parameters, gradient_snapshot)
    _restore_optimizer_state(optimizer, optimizer_snapshot)
    _set_lora_plus_optimizer_ratio(optimizer, base_learning_rate=learning_rate, ratio=lora_plus_ratio)
    optimizer.zero_grad(set_to_none=True)
    loss_context = _loss_context_fixed(
        batch=batch,
        unet=unet,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
        objective=objective,
        noise=candidate_noise,
        timesteps=candidate_timesteps,
    )
    loss_context.loss.backward()
    optimizer.step()
    with torch.no_grad():
        return _evaluate_validation_skeleton(
            skeleton=validation_skeleton,
            unet=unet,
            noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
            step=-1,
        )


def _objective_candidates(
    *,
    current_policy: _TrainingPolicy,
    baseline_report: _ValidationReport,
    validation_skeleton: _ValidationSkeleton,
    noise_scheduler: DDPMScheduler,
) -> list[_ObjectivePolicy]:
    candidates = [_ObjectivePolicy(name="base_mse")]
    gammas = [
        _validation_bucket_gamma(validation_skeleton=validation_skeleton, bucket="mid"),
        5.0,
        _validation_bucket_gamma(
            validation_skeleton=validation_skeleton,
            bucket=max(baseline_report.loss_by_snr_bucket, key=baseline_report.loss_by_snr_bucket.get),
        ),
    ]
    if current_policy.objective.name == "minsnr" and current_policy.objective.gamma is not None:
        gammas.append(current_policy.objective.gamma)
    seen: set[float] = set()
    for gamma in gammas:
        key = round(float(gamma), 8)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(_ObjectivePolicy(name="minsnr", gamma=float(gamma)))
    return candidates


def _validation_bucket_gamma(*, validation_skeleton: _ValidationSkeleton, bucket: str) -> float:
    values = [item.snr for item in validation_skeleton.items if item.snr_bucket == bucket]
    if not values:
        raise LorakitError(f"Validation skeleton has no SNR bucket: {bucket}")
    return float(sum(values) / len(values))


def _objective_label(objective: _ObjectivePolicy) -> str:
    if objective.name == "base_mse":
        return "base_mse"
    if objective.name == "minsnr" and objective.gamma is not None:
        return f"minsnr_gamma_{objective.gamma:.6g}"
    raise LorakitError(f"Cannot label objective: {objective}")


def _ratio_label(ratio: float) -> str:
    return f"ratio_{ratio:.6g}"


def _lora_plus_candidate_ratios(unet: UNet2DConditionModel) -> list[float]:
    scale_a, scale_b = _lora_relative_gradient_scales(unet)
    if scale_a <= 0.0 or scale_b <= 0.0:
        return [1.0]
    ratio = max(0.1, min(10.0, scale_a / max(scale_b, 1e-12)))
    inverse = max(0.1, min(10.0, 1.0 / ratio))
    candidates: list[float] = []
    for value in (1.0, ratio, inverse):
        if not any(abs(value - existing) <= 1e-8 for existing in candidates):
            candidates.append(float(value))
    return candidates


def _lora_relative_gradient_scales(unet: UNet2DConditionModel) -> tuple[float, float]:
    lora_a, lora_b = _partition_lora_parameters(unet)
    return _relative_gradient_scale(lora_a), _relative_gradient_scale(lora_b)


def _partition_lora_parameters(unet: UNet2DConditionModel) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
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


def _relative_gradient_scale(parameters: list[torch.nn.Parameter]) -> float:
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


# ---------------------------------------------------------------------------
# Probes and guardrails
# ---------------------------------------------------------------------------


def _should_probe_contexts(*, global_step: int, sync_gradients: bool) -> bool:
    if not sync_gradients:
        return False
    if global_step < PROBE_INITIAL_STEPS:
        return True
    if global_step % PROBE_EVERY_STEPS != 0:
        return False
    free_bytes = _cuda_free_bytes()
    if free_bytes is not None and free_bytes < PROBE_MIN_FREE_CUDA_BYTES:
        return False
    return True


def _write_probe_status_if_main(
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


def _write_streaming_context_probe_if_main(
    *,
    unet: UNet2DConditionModel,
    optimizer,
    dataset: _DiskCachedLatentDataset,
    batch: dict[str, object],
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
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
) -> _GuardrailDecision:
    parameters = certified_trainable_parameters(unet)
    probe_batches = _build_probe_window(
        dataset=dataset,
        current_batch=batch,
        current_noise=noise,
        current_timesteps=timesteps,
        target_items=PROBE_WINDOW_TARGET_ITEMS,
        unet=unet,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
    )
    context_ids, context_counts, context_losses_tensor, gradients = _collect_probe_context_gradients(
        probe_batches=probe_batches,
        unet=unet,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
        parameters=parameters,
    )
    if not gradients:
        return _default_guardrail_decision()

    probe = probe_from_context_losses(
        context_ids=context_ids,
        context_losses=context_losses_tensor,
        gradients=gradients,
    )
    first_order = candidate_first_order_summaries(
        context_ids=context_ids,
        context_losses=context_losses_tensor,
        gradients=gradients,
        mgda_weights=probe.mgda_lambda,
        context_counts=context_counts,
        step_size=candidate_step_size,
    )
    candidate_gradients = _candidate_gradients_from_probe(
        gradients=gradients,
        mgda_weights=probe.mgda_lambda,
        context_counts=context_counts,
    )
    candidate_certificates, candidate_selection, guardrail_decision = _candidate_step_certificates_for_probe_window(
        unet=unet,
        optimizer=optimizer,
        parameters=parameters,
        probe_batches=probe_batches,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
        context_ids=context_ids,
        old_context_losses=context_losses_tensor,
        candidate_gradients=candidate_gradients,
        step_size=candidate_step_size,
    )

    extra = _probe_extra_metadata(
        batch=batch,
        probe_batches=probe_batches,
        noise_scheduler=noise_scheduler,
        timesteps=timesteps,
        requested_batch_size=requested_batch_size,
        gradient_accumulation=gradient_accumulation,
        training_loss=training_loss,
        cuda_memory_before_probe=cuda_memory_before_probe,
    )
    extra.update(
        {
            "context_example_counts": [int(count) for count in context_counts],
            "candidate_step_size": float(candidate_step_size),
            "candidate_first_order": first_order,
            "candidate_certificates": candidate_certificates,
            "candidate_selection": candidate_selection,
        }
    )

    if should_log:
        payload = probe_to_log_dict(probe, step=step, extra=extra)
        probe_log_path.parent.mkdir(parents=True, exist_ok=True)
        with probe_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
    return guardrail_decision


def _build_probe_window(
    *,
    dataset: _DiskCachedLatentDataset,
    current_batch: dict[str, object],
    current_noise: torch.Tensor,
    current_timesteps: torch.Tensor,
    target_items: int,
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
) -> list[_ProbeBatch]:
    probe_batches = [
        _ProbeBatch(
            batch=current_batch,
            noise=current_noise.detach().cpu(),
            timesteps=current_timesteps.detach().cpu().long(),
            source="training_batch",
        )
    ]
    current_size = int(_batch_tensor(current_batch, "latents").shape[0])
    extra_needed = max(0, int(target_items) - current_size)
    if extra_needed <= 0 or len(dataset) <= 0:
        return probe_batches
    excluded = set(_jsonable_int_list(current_batch.get("record_indices")))
    candidates = [index for index in range(len(dataset)) if index not in excluded]
    random.shuffle(candidates)
    for index in candidates[:extra_needed]:
        item = dataset[index]
        extra_batch = _collate_cached([item])
        extra_noise, extra_timesteps = _sample_probe_noise_and_timesteps(
            batch=extra_batch,
            unet=unet,
            noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
        )
        probe_batches.append(
            _ProbeBatch(
                batch=extra_batch,
                noise=extra_noise,
                timesteps=extra_timesteps,
                source="streamed_probe_item",
            )
        )
    return probe_batches


def _sample_probe_noise_and_timesteps(
    *,
    batch: dict[str, object],
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    latents = _batch_tensor(batch, "latents").to(device=unet.device, dtype=weight_dtype)
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=latents.device,
    ).long()
    return noise.detach().cpu(), timesteps.detach().cpu()


def _collect_probe_context_gradients(
    *,
    probe_batches: list[_ProbeBatch],
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    parameters: list[torch.nn.Parameter],
) -> tuple[tuple[int, ...], list[int], torch.Tensor, list[torch.Tensor]]:
    loss_sums: dict[int, torch.Tensor] = {}
    gradient_sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}

    for probe_batch in probe_batches:
        device_timesteps = probe_batch.timesteps.to(device=unet.device).long()
        local_context_ids = sorted({int(value) for value in device_timesteps.detach().cpu().tolist()})
        for context_id in local_context_ids:
            mask = device_timesteps == context_id
            if not torch.any(mask):
                continue
            example_count = int(mask.sum().item())
            context_loss = _fixed_subset_context_loss(
                batch=probe_batch.batch,
                mask=mask,
                unet=unet,
                noise_scheduler=noise_scheduler,
                weight_dtype=weight_dtype,
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
            flat_gradient = _flatten_grads_from_autograd(parameters=parameters, grads=grads)
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
    context_losses = torch.stack([loss_sums[context_id] / float(counts[context_id]) for context_id in context_ids])
    gradients = [gradient_sums[context_id] / float(counts[context_id]) for context_id in context_ids]
    return context_ids, context_counts, context_losses, gradients


def _candidate_gradients_from_probe(
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


def _candidate_step_certificates_for_probe_window(
    *,
    unet: UNet2DConditionModel,
    optimizer,
    parameters: list[torch.nn.Parameter],
    probe_batches: list[_ProbeBatch],
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    context_ids: tuple[int, ...],
    old_context_losses: torch.Tensor,
    candidate_gradients: dict[str, torch.Tensor],
    step_size: float,
) -> tuple[dict[str, object], dict[str, object], _GuardrailDecision]:
    parameter_snapshot = _snapshot_trainable_parameters(parameters)
    gradient_snapshot = _snapshot_trainable_gradients(parameters)
    optimizer_snapshot = _snapshot_optimizer_state(optimizer)

    certificate_objects: dict[str, StepCertificate] = {}
    results: dict[str, object] = {}
    update_norms: dict[str, float] = {}
    guardrail = _default_guardrail_decision()

    try:
        if optimizer_snapshot is not None:
            _restore_trainable_parameters(parameters, parameter_snapshot)
            _restore_trainable_gradients(parameters, gradient_snapshot)
            _restore_optimizer_state(optimizer, optimizer_snapshot)
            before = _flatten_trainable_parameters(parameters)
            optimizer.step()
            after = _flatten_trainable_parameters(parameters)
            update_norms["adamw_actual"] = float(torch.linalg.vector_norm(after - before).item())
            with torch.no_grad():
                new_context_losses = _evaluate_probe_context_losses(
                    probe_batches=probe_batches,
                    unet=unet,
                    noise_scheduler=noise_scheduler,
                    weight_dtype=weight_dtype,
                    context_ids=context_ids,
                )
            certificate = certify_losses(
                old_context_losses=old_context_losses,
                new_context_losses=new_context_losses,
                backtracks=0,
                step_size=step_size,
                context_ids=context_ids,
            )
            certificate_objects["adamw_actual"] = certificate
            results["adamw_actual"] = certificate_to_log_dict(certificate)
            if not certificate.accepted_bottleneck:
                # Backtrack only if the full actual AdamW step is not certified.
                for backtracks, factor in enumerate(CERTIFIED_BACKTRACK_FACTORS[1:], start=1):
                    _restore_trainable_parameters(parameters, parameter_snapshot)
                    _restore_trainable_gradients(parameters, gradient_snapshot)
                    _restore_optimizer_state(optimizer, optimizer_snapshot)
                    before_bt = _snapshot_trainable_parameters(parameters)
                    optimizer.step()
                    _scale_trainable_update(parameters=parameters, before=before_bt, factor=factor)
                    with torch.no_grad():
                        bt_losses = _evaluate_probe_context_losses(
                            probe_batches=probe_batches,
                            unet=unet,
                            noise_scheduler=noise_scheduler,
                            weight_dtype=weight_dtype,
                            context_ids=context_ids,
                        )
                    bt_cert = certify_losses(
                        old_context_losses=old_context_losses,
                        new_context_losses=bt_losses,
                        backtracks=backtracks,
                        step_size=step_size * factor,
                        context_ids=context_ids,
                    )
                    results[f"adamw_backtrack_{factor:g}"] = certificate_to_log_dict(bt_cert)
                    if bt_cert.accepted_bottleneck:
                        certificate_objects[f"adamw_backtrack_{factor:g}"] = bt_cert
                        break
        else:
            results["adamw_actual"] = {"available": False, "reason": "optimizer_state_snapshot_unavailable"}

        for name, flat_gradient in candidate_gradients.items():
            _restore_trainable_parameters(parameters, parameter_snapshot)
            _restore_trainable_gradients(parameters, gradient_snapshot)
            _apply_flat_gradient_step(parameters=parameters, flat_gradient=flat_gradient, step_size=step_size)
            update_norms[name] = float(torch.linalg.vector_norm(flat_gradient).item() * abs(float(step_size)))
            with torch.no_grad():
                new_context_losses = _evaluate_probe_context_losses(
                    probe_batches=probe_batches,
                    unet=unet,
                    noise_scheduler=noise_scheduler,
                    weight_dtype=weight_dtype,
                    context_ids=context_ids,
                )
            certificate = certify_losses(
                old_context_losses=old_context_losses,
                new_context_losses=new_context_losses,
                backtracks=0,
                step_size=step_size,
                context_ids=context_ids,
            )
            certificate_objects[name] = certificate
            results[name] = certificate_to_log_dict(certificate)

        selection = select_preferred_candidate(certificate_objects, update_norms=update_norms)
        selected = selection.get("selected")
        if selected == "adamw_actual" or (isinstance(selected, str) and selected.startswith("adamw_backtrack_")):
            guardrail = _GuardrailDecision(
                handled_update=False,
                committed_update=str(selected),
                backtracks=int(certificate_objects[str(selected)].backtracks),
                fallback_used=False,
                skipped_update=False,
            )
        elif isinstance(selected, str) and selected in candidate_gradients and certificate_objects[selected].accepted_bottleneck:
            _restore_trainable_parameters(parameters, parameter_snapshot)
            _restore_trainable_gradients(parameters, gradient_snapshot)
            _apply_flat_gradient_step(parameters=parameters, flat_gradient=candidate_gradients[selected], step_size=step_size)
            guardrail = _GuardrailDecision(
                handled_update=True,
                committed_update=selected,
                backtracks=0,
                fallback_used=True,
                skipped_update=False,
            )
        elif certificate_objects and all(not c.accepted_bottleneck for c in certificate_objects.values()):
            _restore_trainable_parameters(parameters, parameter_snapshot)
            _restore_trainable_gradients(parameters, gradient_snapshot)
            guardrail = _GuardrailDecision(
                handled_update=True,
                committed_update="skip_update",
                backtracks=0,
                fallback_used=False,
                skipped_update=True,
            )
        else:
            _restore_trainable_parameters(parameters, parameter_snapshot)
            _restore_trainable_gradients(parameters, gradient_snapshot)

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
            _restore_trainable_parameters(parameters, parameter_snapshot)
            _restore_trainable_gradients(parameters, gradient_snapshot)
            if optimizer_snapshot is not None:
                _restore_optimizer_state(optimizer, optimizer_snapshot)


def _evaluate_probe_context_losses(
    *,
    probe_batches: list[_ProbeBatch],
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    context_ids: tuple[int, ...],
) -> torch.Tensor:
    loss_sums = {context_id: torch.tensor(0.0, dtype=torch.float32) for context_id in context_ids}
    counts = {context_id: 0 for context_id in context_ids}
    for probe_batch in probe_batches:
        device_timesteps = probe_batch.timesteps.to(device=unet.device).long()
        for context_id in context_ids:
            mask = device_timesteps == int(context_id)
            if not torch.any(mask):
                continue
            example_count = int(mask.sum().item())
            loss = _fixed_subset_context_loss(
                batch=probe_batch.batch,
                mask=mask,
                unet=unet,
                noise_scheduler=noise_scheduler,
                weight_dtype=weight_dtype,
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


def _fixed_subset_context_loss(
    *,
    batch: dict[str, object],
    mask: torch.Tensor,
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    latents = _batch_tensor(batch, "latents").to(device=unet.device, dtype=weight_dtype)[mask]
    encoder_hidden_states = _batch_tensor(batch, "encoder_hidden_states").to(device=unet.device, dtype=weight_dtype)[mask]
    local_noise = noise.to(device=unet.device, dtype=weight_dtype)[mask]
    local_timesteps = timesteps.to(device=unet.device).long()[mask]
    noisy_latents = noise_scheduler.add_noise(latents, local_noise, local_timesteps)
    target = _target(
        noise_scheduler=noise_scheduler,
        latents=latents,
        noise=local_noise,
        timesteps=local_timesteps,
    )
    prediction = unet(noisy_latents, local_timesteps, encoder_hidden_states, return_dict=False)[0]
    return denoising_loss_per_example(model_pred=prediction, target=target).mean()


def _flatten_grads_from_autograd(
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


def _flatten_trainable_parameters(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    pieces = [parameter.detach().float().reshape(-1).cpu() for parameter in parameters]
    if not pieces:
        raise LorakitError("No trainable parameters found for candidate evaluation")
    return torch.cat(pieces, dim=0)


def _snapshot_trainable_gradients(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor | None]:
    return [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters]


@torch.no_grad()
def _restore_trainable_gradients(parameters: list[torch.nn.Parameter], snapshot: list[torch.Tensor | None]) -> None:
    for parameter, value in zip(parameters, snapshot, strict=True):
        if value is None:
            parameter.grad = None
        else:
            parameter.grad = value.detach().clone().to(device=parameter.device)


def _snapshot_optimizer_state(optimizer):
    try:
        return copy.deepcopy(optimizer.state_dict())
    except (RuntimeError, TypeError, ValueError):
        return None


def _restore_optimizer_state(optimizer, snapshot) -> None:
    optimizer.load_state_dict(copy.deepcopy(snapshot))


def _snapshot_trainable_parameters(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


@torch.no_grad()
def _restore_trainable_parameters(parameters: list[torch.nn.Parameter], snapshot: list[torch.Tensor]) -> None:
    for parameter, value in zip(parameters, snapshot, strict=True):
        parameter.copy_(value)


@torch.no_grad()
def _scale_trainable_update(
    *,
    parameters: list[torch.nn.Parameter],
    before: list[torch.Tensor],
    factor: float,
) -> None:
    for parameter, previous in zip(parameters, before, strict=True):
        update = parameter.detach() - previous.to(device=parameter.device)
        parameter.copy_(previous.to(device=parameter.device) + update * float(factor))


@torch.no_grad()
def _apply_flat_gradient_step(
    *,
    parameters: list[torch.nn.Parameter],
    flat_gradient: torch.Tensor,
    step_size: float,
) -> None:
    offset = 0
    flat = flat_gradient.detach().float().cpu()
    for parameter in parameters:
        count = parameter.numel()
        chunk = flat[offset : offset + count]
        if chunk.numel() != count:
            raise LorakitError("Flat candidate gradient is shorter than trainable parameter vector")
        update = chunk.reshape(parameter.shape).to(device=parameter.device, dtype=parameter.dtype)
        parameter.add_(update, alpha=-float(step_size))
        offset += count
    if offset != flat.numel():
        raise LorakitError("Flat candidate gradient is longer than trainable parameter vector")


# ---------------------------------------------------------------------------
# Dense-gradient rank growth
# ---------------------------------------------------------------------------


def _maybe_grow_lora_rank(
    *,
    unet: UNet2DConditionModel,
    optimizer,
    scheduler,
    validation_skeleton: _ValidationSkeleton,
    baseline_report: _ValidationReport,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    should_probe: bool,
    base_learning_rate: float,
    lora_plus_ratio: float,
) -> dict[str, object]:
    if not should_probe:
        return {"trigger": "validation_improved", "rank_growth_accepted": False}
    free_cuda_bytes = _cuda_free_bytes()
    if free_cuda_bytes is not None and free_cuda_bytes < RANK_GROWTH_MIN_FREE_CUDA_BYTES:
        return {
            "trigger": "validation_plateau",
            "rank_growth_accepted": False,
            "rank_growth_skipped_memory": True,
            "free_cuda_bytes": free_cuda_bytes,
        }

    proposals = _dense_rank_growth_proposals(
        unet=unet,
        validation_skeleton=validation_skeleton,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
    )
    module_reports = {name: proposal.report for name, proposal in proposals.items()}
    grown: list[tuple[object, str, dict[str, object], list[tuple[str, torch.nn.Parameter]]]] = []

    for name, layer in _iter_lora_layers(unet):
        proposal = proposals.get(name)
        if proposal is None:
            continue
        growth = int(proposal.report.get("rank_growth", 0))
        if growth <= 0:
            continue
        adapter = str(proposal.report["adapter"])
        old_named_parameters = {id(parameter) for _pname, parameter in layer.named_parameters()}
        snapshot = _snapshot_lora_layer(layer=layer, adapter=adapter)
        if _grow_lora_layer_noop(
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
            grown.append((layer, adapter, snapshot, new_named_parameters))

    if not grown:
        return {"trigger": "validation_plateau", "modules": module_reports, "rank_growth_accepted": False}

    grown_report = _evaluate_validation_skeleton(
        skeleton=validation_skeleton,
        unet=unet,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
        step=baseline_report.step,
    )
    accepted = _validation_score_nonworse(baseline=baseline_report, candidate=grown_report)
    if not accepted:
        for layer, adapter, snapshot, _new_named_parameters in grown:
            _restore_lora_layer(layer=layer, adapter=adapter, snapshot=snapshot)
    else:
        groups_before = len(optimizer.param_groups)
        for _layer, _adapter, _snapshot, new_named_parameters in grown:
            if new_named_parameters:
                _add_lora_named_parameters_to_optimizer(
                    optimizer,
                    new_named_parameters,
                    base_learning_rate=base_learning_rate,
                    ratio=lora_plus_ratio,
                )
        _sync_scheduler_param_groups_after_optimizer_growth(
            scheduler=scheduler,
            optimizer=optimizer,
            groups_before=groups_before,
        )

    return {
        "trigger": "validation_plateau",
        "rank_probe_kind": "dense_base_validation_gradient",
        "modules": module_reports,
        "rank_growth_accepted": bool(accepted),
        "validation_loss_max_snr_bucket_delta": float(grown_report.loss_max_snr_bucket - baseline_report.loss_max_snr_bucket),
        "validation_loss_mean_delta": float(grown_report.loss_mean - baseline_report.loss_mean),
    }


def _dense_rank_growth_proposals(
    *,
    unet: UNet2DConditionModel,
    validation_skeleton: _ValidationSkeleton,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
) -> dict[str, _RankGrowthProposal]:
    layers = [(name, layer) for name, layer in _iter_lora_layers(unet)]
    if not layers or len(validation_skeleton.items) < 2:
        return {
            name: _RankGrowthProposal(
                report={"adapter": _first_lora_adapter(layer), "rank_growth": 0, "reason": "insufficient_validation_items"},
                init_a_rows=None,
            )
            for name, layer in layers
        }

    base_weights: list[torch.nn.Parameter] = []
    active_layers: list[tuple[str, object]] = []
    for name, layer in layers:
        base_weight = _lora_base_weight(layer)
        adapter = _first_lora_adapter(layer)
        if adapter is None or base_weight is None:
            continue
        base_weights.append(base_weight)
        active_layers.append((name, layer))

    if not base_weights:
        return {}

    items = list(validation_skeleton.items)
    midpoint = max(1, len(items) // 2)
    first = items[:midpoint]
    second = items[midpoint:] or items[:midpoint]

    gradients_a = _dense_base_gradients_for_items(
        unet=unet,
        items=first,
        base_weights=base_weights,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
    )
    gradients_b = _dense_base_gradients_for_items(
        unet=unet,
        items=second,
        base_weights=base_weights,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
    )

    proposals: dict[str, _RankGrowthProposal] = {}
    for (name, layer), grad_a, grad_b in zip(active_layers, gradients_a, gradients_b, strict=True):
        proposals[name] = _rank_growth_proposal_from_dense_residual(
            name=name,
            layer=layer,
            first_gradient=grad_a,
            second_gradient=grad_b,
        )
    return proposals


def _dense_base_gradients_for_items(
    *,
    unet: UNet2DConditionModel,
    items: list[_ValidationItem],
    base_weights: list[torch.nn.Parameter],
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
) -> list[torch.Tensor]:
    if not items:
        return [torch.zeros_like(weight.detach(), dtype=torch.float32, device="cpu") for weight in base_weights]

    previous_requires_grad = [weight.requires_grad for weight in base_weights]
    for weight in base_weights:
        weight.requires_grad_(True)

    sums = [torch.zeros_like(weight.detach(), dtype=torch.float32, device="cpu") for weight in base_weights]
    try:
        for item in items:
            unet.zero_grad(set_to_none=True)
            loss = _validation_item_loss_tensor(
                item=item,
                unet=unet,
                noise_scheduler=noise_scheduler,
                weight_dtype=weight_dtype,
            )
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
        unet.zero_grad(set_to_none=True)

    scale = 1.0 / float(len(items))
    return [value * scale for value in sums]


def _rank_growth_proposal_from_dense_residual(
    *,
    name: str,
    layer,
    first_gradient: torch.Tensor,
    second_gradient: torch.Tensor,
) -> _RankGrowthProposal:
    adapter = _first_lora_adapter(layer)
    if adapter is None:
        return _RankGrowthProposal(report={"rank_growth": 0, "reason": "no_adapter"}, init_a_rows=None)
    lora_a = _lora_adapter_module(layer.lora_A, adapter)
    lora_b = _lora_adapter_module(layer.lora_B, adapter)
    if lora_a is None or lora_b is None:
        return _RankGrowthProposal(
            report={"adapter": adapter, "rank_growth": 0, "reason": "missing_lora_module"},
            init_a_rows=None,
        )
    if first_gradient.shape != second_gradient.shape:
        return _RankGrowthProposal(
            report={"adapter": adapter, "rank_growth": 0, "reason": "gradient_shape_mismatch"},
            init_a_rows=None,
        )

    signal = 0.5 * (first_gradient.detach().float().cpu() + second_gradient.detach().float().cpu())
    noise = 0.5 * (first_gradient.detach().float().cpu() - second_gradient.detach().float().cpu())
    if signal.ndim != 2:
        return _RankGrowthProposal(
            report={"adapter": adapter, "rank_growth": 0, "reason": "base_gradient_not_matrix"},
            init_a_rows=None,
        )

    signal_values = torch.linalg.svdvals(signal)
    noise_values = torch.linalg.svdvals(noise)
    threshold = _gavish_donoho_unknown_noise_threshold(
        signal_shape=signal.shape,
        noise_singular_values=noise_values,
    )
    residual_rank = int(torch.sum(signal_values > threshold).item())
    growth = min(residual_rank, RANK_GROWTH_MAX_CHANNELS)
    current_rank = int(lora_a.out_features)

    init_a_rows: torch.Tensor | None = None
    if growth > 0:
        try:
            _u, _s, vh = torch.linalg.svd(signal, full_matrices=False)
            init_a_rows = vh[:growth].to(dtype=torch.float32).contiguous()
        except RuntimeError:
            growth = 0
            init_a_rows = None

    report = {
        "adapter": adapter,
        "module": name,
        "current_rank": current_rank,
        "residual_rank": residual_rank,
        "noise_threshold": float(threshold),
        "top_singular_values": [float(value) for value in signal_values[: min(8, int(signal_values.numel()))].tolist()],
        "noise_singular_median": float(torch.median(noise_values).item()) if noise_values.numel() else 0.0,
        "aspect_ratio": float(signal.shape[0] / max(1, signal.shape[1])),
        "rank_growth": int(growth),
        "rank_probe_kind": "dense_base_validation_gradient",
    }
    return _RankGrowthProposal(report=report, init_a_rows=init_a_rows)


def _gavish_donoho_unknown_noise_threshold(
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


def _lora_base_weight(layer) -> torch.nn.Parameter | None:
    base_layer = getattr(layer, "base_layer", None)
    if base_layer is not None and hasattr(base_layer, "weight") and isinstance(base_layer.weight, torch.nn.Parameter):
        return base_layer.weight
    if hasattr(layer, "weight") and isinstance(layer.weight, torch.nn.Parameter):
        return layer.weight
    return None


def _iter_lora_layers(unet: UNet2DConditionModel):
    for name, module in unet.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            yield name, module


def _first_lora_adapter(layer) -> str | None:
    lora_a = getattr(layer, "lora_A", None)
    if hasattr(lora_a, "keys"):
        keys = list(lora_a.keys())
        return str(keys[0]) if keys else None
    return "default" if lora_a is not None else None


def _lora_adapter_module(container, adapter: str):
    if hasattr(container, "__getitem__"):
        try:
            return container[adapter]
        except (KeyError, TypeError):
            return None
    return container


def _grow_lora_layer_noop(
    *,
    layer,
    adapter: str,
    growth: int,
    init_a_rows: torch.Tensor | None = None,
) -> bool:
    if growth <= 0:
        return False
    lora_a = _lora_adapter_module(layer.lora_A, adapter)
    lora_b = _lora_adapter_module(layer.lora_B, adapter)
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
            usable = min(int(growth), rows.shape[0], new_a.weight.shape[1], rows.shape[1])
            if usable > 0 and rows.shape[1] == new_a.weight.shape[1]:
                new_a.weight[old_rank : old_rank + usable].copy_(rows[:usable])
    _set_lora_adapter_module(layer.lora_A, adapter, new_a)
    _set_lora_adapter_module(layer.lora_B, adapter, new_b)
    if hasattr(layer, "r") and isinstance(layer.r, dict):
        layer.r[adapter] = new_rank
    if hasattr(layer, "lora_alpha") and isinstance(layer.lora_alpha, dict):
        layer.lora_alpha[adapter] = new_rank
    if hasattr(layer, "scaling") and isinstance(layer.scaling, dict):
        alpha = float(layer.lora_alpha.get(adapter, new_rank)) if hasattr(layer, "lora_alpha") else float(new_rank)
        layer.scaling[adapter] = alpha / float(new_rank)
    return True


def _snapshot_lora_layer(*, layer, adapter: str) -> dict[str, object]:
    lora_a = _lora_adapter_module(layer.lora_A, adapter)
    lora_b = _lora_adapter_module(layer.lora_B, adapter)
    if lora_a is None or lora_b is None:
        raise LorakitError(f"Cannot snapshot missing LoRA adapter: {adapter}")
    return {
        "lora_A": copy.deepcopy(lora_a),
        "lora_B": copy.deepcopy(lora_b),
        "r": copy.deepcopy(getattr(layer, "r", {}).get(adapter) if hasattr(layer, "r") else None),
        "lora_alpha": copy.deepcopy(getattr(layer, "lora_alpha", {}).get(adapter) if hasattr(layer, "lora_alpha") else None),
        "scaling": copy.deepcopy(getattr(layer, "scaling", {}).get(adapter) if hasattr(layer, "scaling") else None),
    }


def _restore_lora_layer(*, layer, adapter: str, snapshot: dict[str, object]) -> None:
    _set_lora_adapter_module(layer.lora_A, adapter, snapshot["lora_A"])
    _set_lora_adapter_module(layer.lora_B, adapter, snapshot["lora_B"])
    if hasattr(layer, "r") and isinstance(layer.r, dict):
        layer.r[adapter] = snapshot["r"]
    if hasattr(layer, "lora_alpha") and isinstance(layer.lora_alpha, dict):
        layer.lora_alpha[adapter] = snapshot["lora_alpha"]
    if hasattr(layer, "scaling") and isinstance(layer.scaling, dict):
        layer.scaling[adapter] = snapshot["scaling"]


def _set_lora_adapter_module(container, adapter: str, module: torch.nn.Module) -> None:
    if hasattr(container, "__setitem__"):
        container[adapter] = module
        return
    raise LorakitError("LoRA adapter container does not support replacement")


# ---------------------------------------------------------------------------
# Logging and summaries
# ---------------------------------------------------------------------------


def _probe_extra_metadata(
    *,
    batch: dict[str, object],
    probe_batches: list[_ProbeBatch],
    noise_scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
    requested_batch_size: int,
    gradient_accumulation: int,
    training_loss: torch.Tensor,
    cuda_memory_before_probe: dict[str, int] | None,
) -> dict[str, object]:
    latents = _batch_tensor(batch, "latents")
    encoder_hidden_states = _batch_tensor(batch, "encoder_hidden_states")
    detached_timesteps = timesteps.detach().cpu()
    snr = _snr_for_timesteps(noise_scheduler=noise_scheduler, timesteps=detached_timesteps.long()).detach().float().cpu()

    window_timesteps: list[int] = []
    window_snr_values: list[float] = []
    window_record_indices: list[int] = []
    window_images: list[str] = []
    window_sources: list[str] = []
    window_latent_item_shapes: list[list[int]] = []
    for probe_batch in probe_batches:
        probe_latents = _batch_tensor(probe_batch.batch, "latents")
        probe_timesteps = probe_batch.timesteps.detach().cpu().long()
        probe_snr = _snr_for_timesteps(noise_scheduler=noise_scheduler, timesteps=probe_timesteps).detach().float().cpu()
        batch_size = int(probe_latents.shape[0])
        window_timesteps.extend(int(item) for item in probe_timesteps.tolist())
        window_snr_values.extend(float(item) for item in probe_snr.tolist())
        window_record_indices.extend(_jsonable_int_list(probe_batch.batch.get("record_indices")))
        window_images.extend(_jsonable_str_list(probe_batch.batch.get("images")))
        window_sources.extend([probe_batch.source] * batch_size)
        window_latent_item_shapes.extend([[int(item) for item in probe_latents.shape[1:]]] * batch_size)

    window_snr = torch.tensor(window_snr_values, dtype=torch.float32)
    return {
        "probe_version": PROBE_VERSION,
        "microbatch_size": int(latents.shape[0]),
        "requested_batch_size": int(requested_batch_size),
        "gradient_accumulation": int(gradient_accumulation),
        "latent_batch_shape": [int(item) for item in latents.shape],
        "latent_item_shape": [int(item) for item in latents.shape[1:]],
        "encoder_hidden_state_batch_shape": [int(item) for item in encoder_hidden_states.shape],
        "encoder_hidden_state_item_shape": [int(item) for item in encoder_hidden_states.shape[1:]],
        "record_indices": _jsonable_int_list(batch.get("record_indices")),
        "images": _jsonable_str_list(batch.get("images")),
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
        "cuda_memory_after_probe": _cuda_memory_snapshot(),
    }


def _write_policy_challenger_report_if_main(
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


def _write_probe_summary(*, probe_log_path: Path, summary_path: Path) -> None:
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
        "best_checkpoint_step": _best_validation_step(validation_reports),
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


def _best_validation_step(validation_reports: list[dict[str, object]]) -> int | None:
    improved = [report for report in validation_reports if bool(report.get("best_checkpoint_improved", False))]
    if not improved:
        return None
    return int(improved[-1]["step"])


def _save_named_lora_weights(*, unet: torch.nn.Module, output_dir: Path, filename: str) -> None:
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


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _cuda_free_bytes() -> int | None:
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def _cuda_memory_snapshot() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return {
        "free_cuda_bytes": int(free),
        "total_cuda_bytes": int(total),
        "allocated_cuda_bytes": int(torch.cuda.memory_allocated()),
        "reserved_cuda_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_cuda_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _jsonable_int_list(value: object) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, list):
        return [int(item) for item in value]
    return []


def _jsonable_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []
