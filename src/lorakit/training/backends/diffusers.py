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

import gc
import shutil
from importlib.util import find_spec
from pathlib import Path
from typing import Final

import torch
from accelerate import Accelerator
from peft import LoraConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler

from lorakit.errors import LorakitError
from lorakit.training.backends._artifacts import (
    save_named_lora_weights as _save_named_lora_weights,
    write_policy_challenger_report_if_main as _write_policy_challenger_report_if_main,
    write_probe_summary as _write_probe_summary,
)
from lorakit.training.backends._cache import (
    CACHE_DIR_NAME,
    CACHE_MANIFEST_NAME,
    DiskCachedLatentDataset as _DiskCachedLatentDataset,
    LatentShapeBatchSampler as _LatentShapeBatchSampler,
    cache_encoder_hidden_states as _cache_encoder_hidden_states,
    cache_latents as _cache_latents,
    cache_records as _cache_records,
    cleanup_after_cuda_oom as _cleanup_after_cuda_oom,
    collate_cached as _collate_cached,
    is_cuda_oom as _is_cuda_oom,
    load_rows as _load_rows,
    write_cache_manifest as _write_cache_manifest,
)
from lorakit.training.backends._challengers import (
    FrozenBatch as _FrozenBatch,
    PolicyChallenger as _PolicyChallenger,
)
from lorakit.training.backends._evaluator import DiffusionValidator as _DiffusionValidator
from lorakit.training.backends._loss import (
    loss_context as _loss_context,
)
from lorakit.training.backends._optimizer import (
    optimizer as _optimizer,
    set_lora_plus_optimizer_ratio as _set_lora_plus_optimizer_ratio,
)
from lorakit.training.backends._policy import (
    default_training_policy as _default_training_policy,
)
from lorakit.training.backends._probes import (
    ContextProber as _ContextProber,
    ProbeConfig as _ProbeConfig,
    default_guardrail_decision as _default_guardrail_decision,
    should_probe_contexts as _should_probe_contexts,
    write_probe_status_if_main as _write_probe_status_if_main,
)
from lorakit.training.backends._rank import (
    maybe_grow_lora_rank as _maybe_grow_lora_rank,
)
from lorakit.training.backends._validation import (
    BestCheckpoint as _BestCheckpoint,
    build_validation_skeleton as _build_validation_skeleton,
    split_train_validation_records as _split_train_validation_records,
    write_validation_report_if_main as _write_validation_report_if_main,
)
from lorakit.training.backends.types import BackendResult, BackendSpec


LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]
LORA_WEIGHTS_NAME = "pytorch_lora_weights.safetensors"
BEST_LORA_WEIGHTS_NAME = "best_lora_weights.safetensors"
MAX_GRAD_NORM = 1.0
LR_WARMUP_STEPS = 0

VALIDATION_EVERY_PROBES: Final = 1

PROBE_LOG_NAME = "context-probes.jsonl"
PROBE_SUMMARY_NAME = "context-probes-summary.json"


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
    validator = _DiffusionValidator(
        skeleton=validation_skeleton,
        unet=unet,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
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
    probe_config = _ProbeConfig()
    prober = _ContextProber(
        unet=unet,
        optimizer=optimizer,
        dataset=dataset,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
        config=probe_config,
        cuda_memory_snapshot=_cuda_memory_snapshot,
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
                    free_cuda_bytes=_cuda_free_bytes(),
                    config=probe_config,
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
                            "probe_version": probe_config.version,
                            "microbatch_size": int(loss_context.noise.shape[0]),
                            "requested_batch_size": int(spec.batch_size),
                            "gradient_accumulation": int(spec.gradient_accumulation),
                            "active_objective": training_policy.objective.label,
                            "active_lora_plus_ratio": float(training_policy.lora_plus_ratio),
                        },
                    )
                    try:
                        guardrail_decision = prober.run(
                            batch=batch,
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
                                "probe_version": probe_config.version,
                                "requested_batch_size": int(spec.batch_size),
                                "gradient_accumulation": int(spec.gradient_accumulation),
                            },
                        )
                elif accelerator.sync_gradients and accelerator.is_local_main_process:
                    free_cuda_bytes = _cuda_free_bytes()
                    if free_cuda_bytes is not None and free_cuda_bytes < probe_config.min_free_cuda_bytes:
                        _write_probe_status_if_main(
                            probe_log_path=probe_log_path,
                            step=global_step,
                            should_log=True,
                            status="skipped_low_cuda_memory",
                            free_cuda_bytes=free_cuda_bytes,
                            extra={
                                "probe_version": probe_config.version,
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
                        validation_report = validator.score(step=global_step)
                        improved, best_checkpoint = best_checkpoint.consider(validation_report)
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

                        challenger = _PolicyChallenger(
                            unet=unet,
                            optimizer=optimizer,
                            validator=validator,
                            noise_scheduler=noise_scheduler,
                            weight_dtype=weight_dtype,
                            learning_rate=spec.learning_rate,
                        )
                        training_policy, challenger_log = challenger.choose(
                            current_policy=training_policy,
                            frozen_batch=_FrozenBatch(
                                batch=batch,
                                noise=loss_context.noise,
                                timesteps=loss_context.timesteps,
                            ),
                            baseline=validation_report,
                        )
                        challenger_log["rank_probe"] = _maybe_grow_lora_rank(
                            unet=unet,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            validation_skeleton=validator.skeleton,
                            baseline_report=validation_report,
                            noise_scheduler=noise_scheduler,
                            weight_dtype=weight_dtype,
                            should_probe=not improved,
                            base_learning_rate=spec.learning_rate,
                            lora_plus_ratio=training_policy.lora_plus_ratio,
                            validation_loss_for_item=validator.loss,
                            validation_loss_tensor_for_item=validator.loss_tensor,
                            free_cuda_bytes=_cuda_free_bytes(),
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
