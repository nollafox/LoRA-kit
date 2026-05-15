"""Diffusers LoRA training backend."""

import copy
import gc
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from accelerate import Accelerator
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers

from lorakit.errors import LorakitError
from lorakit.manifest import MANIFEST_NAME, read_manifest
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
VAE_DOWNSAMPLE_FACTOR: Final = 8
VALIDATION_FRACTION: Final = 0.08
VALIDATION_MAX_ITEMS: Final = 32
VALIDATION_EVERY_PROBES: Final = 1
VALIDATION_SNR_BUCKETS: Final = ("high", "mid", "low")
VALIDATION_SCORE_ABSOLUTE_TOLERANCE: Final = 1e-7
VALIDATION_SCORE_RELATIVE_TOLERANCE: Final = 1e-5
CACHE_ENCODING_MAX_BATCH_SIZE: Final = 4
CACHE_ENCODING_MIN_BATCH_SIZE: Final = 1
CACHE_DIR_NAME: Final = "tensor-cache"
CACHE_MANIFEST_NAME: Final = "manifest.json"
LATENT_CACHE_MODE: Final = "posterior_mode"
PROBE_LOG_NAME = "context-probes.jsonl"
PROBE_SUMMARY_NAME = "context-probes-summary.json"
PROBE_INITIAL_STEPS: Final = 3
PROBE_EVERY_STEPS: Final = 25
PROBE_MIN_FREE_CUDA_BYTES: Final = 500_000_000
PROBE_WINDOW_TARGET_ITEMS: Final = 4
PROBE_VERSION: Final = 4
CERTIFIED_BACKTRACK_FACTORS: Final = (1.0, 0.5, 0.25, 0.125)
HASH_CHUNK_SIZE_BYTES: Final = 1024 * 1024


def _require_training_acceleration_packages() -> None:
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise LorakitError(
            "bitsandbytes is required for Lorakit's Diffusers backend. "
            "Install it with: poetry run pip install bitsandbytes"
        ) from exc


def _enable_memory_efficient_attention(unet: UNet2DConditionModel) -> str:
    try:
        unet.enable_xformers_memory_efficient_attention()
        return "xformers"
    except Exception:
        # Fall back to PyTorch 2.x SDPA/default attention.
        return "torch"


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


def train(spec: BackendSpec) -> BackendResult:
    _require_training_acceleration_packages()
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
        batch_sampler=_LatentShapeBatchSampler(
            records=train_records,
            batch_size=spec.batch_size,
        ),
        collate_fn=_collate_cached,
        num_workers=0,
        pin_memory=False,
    )
    trainable_parameters = [
        parameter for parameter in unet.parameters() if parameter.requires_grad
    ]
    optimizer = _optimizer(trainable_parameters, spec.learning_rate)
    scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=LR_WARMUP_STEPS,
        num_training_steps=spec.steps * accelerator.num_processes,
    )
    unet, optimizer, dataloader, scheduler = accelerator.prepare(
        unet,
        optimizer,
        dataloader,
        scheduler,
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
                loss_context = _loss_context(
                    batch=batch,
                    unet=unet,
                    noise_scheduler=noise_scheduler,
                    weight_dtype=weight_dtype,
                    objective=training_policy.objective,
                )
                loss = loss_context.loss
                accelerator.backward(loss)
                _apply_lora_plus_gradient_ratio(
                    unet=unet,
                    ratio=training_policy.lora_plus_ratio,
                )
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [parameter for parameter in unet.parameters() if parameter.requires_grad],
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
                    except Exception as exc:
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
                            unet=unet,
                            optimizer=optimizer,
                            validation_skeleton=validation_skeleton,
                            baseline_report=validation_report,
                            noise_scheduler=noise_scheduler,
                            weight_dtype=weight_dtype,
                            learning_rate=spec.learning_rate,
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


@dataclass(frozen=True)
class _TrainingRow:
    image: str
    caption: str
    image_path: Path
    image_sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class _CacheRecord:
    image: str
    caption_sha256: str
    image_sha256: str
    latent_path: Path
    encoder_hidden_state_path: Path
    latent_shape: tuple[int, ...]
    encoder_hidden_state_shape: tuple[int, ...]


@dataclass(frozen=True)
class _ValidationItem:
    record: _CacheRecord
    timestep: int
    noise_seed: int
    snr: float
    snr_bucket: str


@dataclass(frozen=True)
class _ValidationSkeleton:
    items: tuple[_ValidationItem, ...]


@dataclass(frozen=True)
class _ValidationReport:
    step: int
    loss_mean: float
    loss_max_snr_bucket: float
    loss_by_snr_bucket: dict[str, float]
    item_count: int


@dataclass(frozen=True)
class _BestCheckpoint:
    path: Path
    step: int | None
    loss_max_snr_bucket: float | None
    loss_mean: float | None

    @classmethod
    def empty(cls, path: Path) -> "_BestCheckpoint":
        return cls(
            path=path,
            step=None,
            loss_max_snr_bucket=None,
            loss_mean=None,
        )


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


def _default_guardrail_decision() -> _GuardrailDecision:
    return _GuardrailDecision(
        handled_update=False,
        committed_update="adamw_actual",
        backtracks=0,
        fallback_used=False,
        skipped_update=False,
    )


def _default_training_policy() -> _TrainingPolicy:
    return _TrainingPolicy(
        objective=_ObjectivePolicy(name="base_mse"),
        lora_plus_ratio=1.0,
    )


class _DiskCachedLatentDataset(Dataset):
    def __init__(
        self,
        *,
        records: list[_CacheRecord],
    ):
        if not records:
            raise LorakitError("Tensor cache has no records")
        _assert_cache_records_are_consistent(records)
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self._records[index]
        return {
            "latents": _load_tensor(record.latent_path, expected_shape=record.latent_shape),
            "encoder_hidden_states": _load_tensor(
                record.encoder_hidden_state_path,
                expected_shape=record.encoder_hidden_state_shape,
            ),
            "record_index": index,
            "image": record.image,
            "latent_shape": record.latent_shape,
            "encoder_hidden_state_shape": record.encoder_hidden_state_shape,
        }


class _LatentShapeBatchSampler(BatchSampler):
    def __init__(
        self,
        *,
        records: list[_CacheRecord],
        batch_size: int,
    ):
        if batch_size <= 0:
            raise LorakitError("Batch size must be greater than zero")
        buckets: dict[tuple[int, ...], list[int]] = {}
        for index, record in enumerate(records):
            buckets.setdefault(record.latent_shape, []).append(index)
        if not buckets:
            raise LorakitError("Tensor cache has no latent-shape buckets")
        self._buckets = buckets
        self._batch_size = batch_size

    def __iter__(self):
        bucket_keys = list(self._buckets)
        random.shuffle(bucket_keys)
        for bucket_key in bucket_keys:
            indices = list(self._buckets[bucket_key])
            random.shuffle(indices)
            for start in range(0, len(indices), self._batch_size):
                yield indices[start : start + self._batch_size]

    def __len__(self) -> int:
        return sum(
            (len(indices) + self._batch_size - 1) // self._batch_size
            for indices in self._buckets.values()
        )


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


def _has_meta_tensors(module: torch.nn.Module) -> bool:
    return any(
        param.device.type == "meta" for param in module.parameters()
    ) or any(buffer.device.type == "meta" for buffer in module.buffers())


def _move_module_to_device(
    module: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.nn.Module:
    if _has_meta_tensors(module):
        module.to_empty(device=device)
        if dtype is not None:
            module.to(dtype)
    else:
        if dtype is not None:
            module.to(device, dtype)
        else:
            module.to(device)
    return module


def _load_rows(prepared_dir: Path) -> list[_TrainingRow]:
    rows = read_manifest(prepared_dir / MANIFEST_NAME)
    if not rows:
        raise LorakitError(f"Prepared dataset has no training rows: {prepared_dir}")
    loaded_rows: list[_TrainingRow] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise LorakitError(f"Prepared manifest row {index} must be an object")
        image = row.get("image")
        caption = row.get("caption")
        if not isinstance(image, str) or image == "":
            raise LorakitError(f"Prepared manifest row {index} is missing image path")
        if not isinstance(caption, str):
            raise LorakitError(f"Prepared manifest row {index} is missing caption")
        image_path = prepared_dir / image
        if not image_path.exists() or not image_path.is_file():
            raise LorakitError(f"Prepared manifest row {index} image does not exist: {image_path}")
        with Image.open(image_path) as opened_image:
            width, height = opened_image.size
        if width <= 0 or height <= 0:
            raise LorakitError(f"Prepared image has invalid dimensions: {image_path}")
        if width % VAE_DOWNSAMPLE_FACTOR != 0 or height % VAE_DOWNSAMPLE_FACTOR != 0:
            raise LorakitError(
                "Prepared image dimensions must be divisible by "
                f"{VAE_DOWNSAMPLE_FACTOR}: {image_path} is {width}x{height}"
            )
        loaded_rows.append(
            _TrainingRow(
                image=image,
                caption=caption,
                image_path=image_path,
                image_sha256=_file_sha256(image_path),
                width=width,
                height=height,
            )
        )
    return loaded_rows


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


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "cuda out of memory" in str(exc).lower()


def _cleanup_after_cuda_oom() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _adaptive_cuda_batches(
    *,
    indices: list[int],
    initial_batch_size: int,
    encode_batch,
) -> None:
    batch_size = max(CACHE_ENCODING_MIN_BATCH_SIZE, int(initial_batch_size))
    cursor = 0
    while cursor < len(indices):
        current = indices[cursor : cursor + batch_size]
        try:
            encode_batch(current)
            cursor += len(current)
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            _cleanup_after_cuda_oom()
            if batch_size <= CACHE_ENCODING_MIN_BATCH_SIZE:
                raise LorakitError(
                    "CUDA out of memory while encoding a single cache item. "
                    "Try closing other GPU processes or preparing at a smaller resolution."
                ) from exc
            batch_size = max(CACHE_ENCODING_MIN_BATCH_SIZE, batch_size // 2)


@torch.no_grad()
def _cache_encoder_hidden_states(
    *,
    rows: list[_TrainingRow],
    cache_dir: Path,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: torch.device,
    dtype: torch.dtype,
) -> list[Path]:
    text_cache_dir = cache_dir / "text"
    text_cache_dir.mkdir(parents=True, exist_ok=True)
    text_encoder.requires_grad_(False)
    text_encoder.to(device=device, dtype=dtype)
    text_encoder.eval()
    cached: list[Path | None] = [None] * len(rows)
    progress = tqdm(total=len(rows), desc="Encoding captions")

    def encode_batch(batch_indices: list[int]) -> None:
        tokens = tokenizer(
            [rows[index].caption for index in batch_indices],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        hidden = text_encoder(tokens, return_dict=False)[0]
        for index, tensor in zip(batch_indices, hidden, strict=True):
            path = text_cache_dir / f"{index:08d}.pt"
            _save_tensor(path, tensor.detach().to(dtype=dtype).cpu())
            cached[index] = path
        progress.update(len(batch_indices))
        del tokens
        del hidden

    try:
        _adaptive_cuda_batches(
            indices=list(range(len(rows))),
            initial_batch_size=CACHE_ENCODING_MAX_BATCH_SIZE,
            encode_batch=encode_batch,
        )
    finally:
        progress.close()

    if any(path is None for path in cached):
        raise LorakitError("Text embedding cache did not encode every row")
    return [path for path in cached if path is not None]


@torch.no_grad()
def _cache_latents(
    *,
    rows: list[_TrainingRow],
    cache_dir: Path,
    vae: AutoencoderKL,
    device: torch.device,
    dtype: torch.dtype,
) -> list[Path]:
    latent_cache_dir = cache_dir / "latents"
    latent_cache_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    vae.requires_grad_(False)
    vae.to(device=device, dtype=dtype)
    vae.eval()
    cached: list[Path | None] = [None] * len(rows)
    shape_buckets = _pixel_shape_buckets(rows)
    progress = tqdm(total=len(rows), desc="Encoding latents")

    def encode_batch(batch_indices: list[int]) -> None:
        pixel_batches: list[torch.Tensor] = []
        for index in batch_indices:
            with Image.open(rows[index].image_path) as image:
                pixel_batches.append(transform(image.convert("RGB")))
        pixel_values = torch.stack(pixel_batches).to(device=device, dtype=dtype)
        latent_distribution = vae.encode(pixel_values).latent_dist
        latents = _deterministic_latents(latent_distribution)
        latents = latents * vae.config.scaling_factor
        for index, tensor in zip(batch_indices, latents, strict=True):
            path = latent_cache_dir / f"{index:08d}.pt"
            _save_tensor(path, tensor.detach().to(dtype=dtype).cpu())
            cached[index] = path
        progress.update(len(batch_indices))
        del pixel_batches
        del pixel_values
        del latents

    try:
        for indices in shape_buckets.values():
            _adaptive_cuda_batches(
                indices=indices,
                initial_batch_size=CACHE_ENCODING_MAX_BATCH_SIZE,
                encode_batch=encode_batch,
            )
    finally:
        progress.close()

    if any(path is None for path in cached):
        raise LorakitError("Latent cache did not encode every row")
    return [path for path in cached if path is not None]


def _pixel_shape_buckets(rows: list[_TrainingRow]) -> dict[tuple[int, int], list[int]]:
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(rows):
        buckets.setdefault((row.width, row.height), []).append(index)
    return buckets


def _pixel_shape_cache_batches(rows: list[_TrainingRow]) -> list[list[int]]:
    buckets = _pixel_shape_buckets(rows)
    batches: list[list[int]] = []
    for indices in buckets.values():
        for start in range(0, len(indices), CACHE_ENCODING_MAX_BATCH_SIZE):
            batches.append(indices[start : start + CACHE_ENCODING_MAX_BATCH_SIZE])
    return batches


def _deterministic_latents(latent_distribution) -> torch.Tensor:
    if hasattr(latent_distribution, "mode"):
        mode = latent_distribution.mode()
        if isinstance(mode, torch.Tensor):
            return mode
    mean = getattr(latent_distribution, "mean", None)
    if isinstance(mean, torch.Tensor):
        return mean
    raise LorakitError("VAE latent distribution does not expose mode() or mean")


def _cache_records(
    *,
    rows: list[_TrainingRow],
    latent_paths: list[Path],
    encoder_hidden_state_paths: list[Path],
) -> list[_CacheRecord]:
    if len(rows) != len(latent_paths):
        raise LorakitError("Latent cache length does not match prepared manifest length")
    if len(rows) != len(encoder_hidden_state_paths):
        raise LorakitError("Text embedding cache length does not match prepared manifest length")
    records: list[_CacheRecord] = []
    for row, latent_path, hidden_path in zip(
        rows,
        latent_paths,
        encoder_hidden_state_paths,
        strict=True,
    ):
        latent = _load_tensor(latent_path)
        hidden = _load_tensor(hidden_path)
        records.append(
            _CacheRecord(
                image=row.image,
                caption_sha256=_text_sha256(row.caption),
                image_sha256=row.image_sha256,
                latent_path=latent_path,
                encoder_hidden_state_path=hidden_path,
                latent_shape=tuple(latent.shape),
                encoder_hidden_state_shape=tuple(hidden.shape),
            )
        )
    _assert_cache_records_are_consistent(records)
    return records


def _split_train_validation_records(
    records: list[_CacheRecord],
) -> tuple[list[_CacheRecord], list[_CacheRecord]]:
    if len(records) < 2:
        return records, []
    validation_count = min(
        VALIDATION_MAX_ITEMS,
        max(1, int(round(len(records) * VALIDATION_FRACTION))),
    )
    validation_candidates = sorted(
        records,
        key=_validation_split_key,
    )[:validation_count]
    validation_hashes = {
        record.image_sha256
        for record in validation_candidates
    }
    train_records = [
        record
        for record in records
        if record.image_sha256 not in validation_hashes
    ]
    validation_records = [
        record
        for record in records
        if record.image_sha256 in validation_hashes
    ]
    if not train_records:
        return records, []
    return train_records, validation_records


def _validation_split_key(record: _CacheRecord) -> str:
    return hashlib.sha256(
        f"lorakit-validation-v1:{record.image_sha256}".encode("utf-8")
    ).hexdigest()


def _build_validation_skeleton(
    *,
    records: list[_CacheRecord],
    noise_scheduler: DDPMScheduler,
) -> _ValidationSkeleton:
    if not records:
        return _ValidationSkeleton(items=())
    bucket_timesteps = _validation_bucket_timesteps(
        total_timesteps=int(noise_scheduler.config.num_train_timesteps),
    )
    timesteps = [
        timestep
        for _record in records
        for timestep in bucket_timesteps.values()
    ]
    snr_values = _snr_for_timesteps(
        noise_scheduler=noise_scheduler,
        timesteps=torch.tensor(timesteps, dtype=torch.long),
    ).detach().float().cpu()
    bucket_names = [
        bucket
        for _record in records
        for bucket in bucket_timesteps
    ]
    items = tuple(
        _ValidationItem(
            record=record,
            timestep=int(timestep),
            noise_seed=_validation_noise_seed(record=record, snr_bucket=bucket),
            snr=float(snr),
            snr_bucket=bucket,
        )
        for record, timestep, bucket, snr in zip(
            [
                record
                for record in records
                for _bucket in bucket_timesteps
            ],
            timesteps,
            bucket_names,
            snr_values.tolist(),
            strict=True,
        )
    )
    return _ValidationSkeleton(items=items)


def _validation_bucket_timesteps(*, total_timesteps: int) -> dict[str, int]:
    if total_timesteps <= 0:
        raise LorakitError("Noise scheduler must expose at least one timestep")
    return {
        "high": _validation_timestep_at_fraction(
            fraction=0.10,
            total_timesteps=total_timesteps,
        ),
        "mid": _validation_timestep_at_fraction(
            fraction=0.50,
            total_timesteps=total_timesteps,
        ),
        "low": _validation_timestep_at_fraction(
            fraction=0.90,
            total_timesteps=total_timesteps,
        ),
    }


def _validation_timestep_at_fraction(*, fraction: float, total_timesteps: int) -> int:
    if total_timesteps <= 0:
        raise LorakitError("Noise scheduler must expose at least one timestep")
    if fraction < 0.0 or fraction > 1.0:
        raise LorakitError(f"Validation timestep fraction must be within [0, 1]: {fraction}")
    return min(
        total_timesteps - 1,
        max(0, int(round(float(total_timesteps - 1) * fraction))),
    )


def _validation_noise_seed(*, record: _CacheRecord, snr_bucket: str) -> int:
    digest = hashlib.sha256(
        f"lorakit-validation-noise-v1:{record.image_sha256}:{snr_bucket}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16) % (2**31)


def _snr_buckets(snr_values: torch.Tensor) -> list[str]:
    if snr_values.ndim != 1:
        raise LorakitError("Validation SNR values must be rank-1")
    if snr_values.numel() == 0:
        return []
    ordered = sorted(float(value) for value in snr_values.tolist())
    low_threshold = ordered[int((len(ordered) - 1) / 3)]
    high_threshold = ordered[int(2 * (len(ordered) - 1) / 3)]
    buckets: list[str] = []
    for value in snr_values.tolist():
        current = float(value)
        if current <= low_threshold:
            buckets.append("low")
        elif current <= high_threshold:
            buckets.append("mid")
        else:
            buckets.append("high")
    return buckets


def _assert_cache_records_are_consistent(records: list[_CacheRecord]) -> None:
    first = records[0]
    for record in records:
        if not record.latent_path.exists():
            raise LorakitError(f"Missing latent cache tensor: {record.latent_path}")
        if not record.encoder_hidden_state_path.exists():
            raise LorakitError(f"Missing text cache tensor: {record.encoder_hidden_state_path}")
        if record.encoder_hidden_state_shape != first.encoder_hidden_state_shape:
            raise LorakitError(
                "Text cache tensor shape mismatch: "
                f"expected {first.encoder_hidden_state_shape}, "
                f"got {record.encoder_hidden_state_shape} for {record.image}"
            )


def _write_cache_manifest(
    path: Path,
    records: list[_CacheRecord],
    *,
    latent_scaling_factor: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "count": len(records),
        "vae_latent_cache": {
            "mode": LATENT_CACHE_MODE,
            "scaling_factor": float(latent_scaling_factor),
        },
        "latent_shapes": [list(shape) for shape in sorted({record.latent_shape for record in records})],
        "encoder_hidden_state_shape": list(records[0].encoder_hidden_state_shape),
        "records": [
            {
                "image": record.image,
                "image_sha256": record.image_sha256,
                "caption_sha256": record.caption_sha256,
                "latent_shape": list(record.latent_shape),
                "latent": str(record.latent_path.relative_to(path.parent)),
                "encoder_hidden_state": str(
                    record.encoder_hidden_state_path.relative_to(path.parent)
                ),
            }
            for record in records
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def _load_tensor(
    path: Path,
    expected_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise LorakitError(f"Cached tensor is not a torch.Tensor: {path}")
    if expected_shape is not None and tuple(tensor.shape) != expected_shape:
        raise LorakitError(
            f"Cached tensor shape changed for {path}: "
            f"expected {expected_shape}, got {tuple(tensor.shape)}"
        )
    return tensor


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_SIZE_BYTES)
            if chunk == b"":
                break
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _collate_cached(examples: list[dict[str, object]]) -> dict[str, object]:
    latents = [example["latents"] for example in examples]
    encoder_hidden_states = [example["encoder_hidden_states"] for example in examples]

    if not all(isinstance(value, torch.Tensor) for value in latents):
        raise LorakitError("Cached latent batch contains non-tensor values")
    if not all(isinstance(value, torch.Tensor) for value in encoder_hidden_states):
        raise LorakitError("Cached text embedding batch contains non-tensor values")

    return {
        "latents": torch.stack(latents).to(memory_format=torch.contiguous_format),
        "encoder_hidden_states": torch.stack(encoder_hidden_states).to(
            memory_format=torch.contiguous_format
        ),
        "record_indices": [int(example["record_index"]) for example in examples],
        "images": [str(example["image"]) for example in examples],
        "latent_shape": list(latents[0].shape),
        "encoder_hidden_state_shape": list(encoder_hidden_states[0].shape),
    }


def _optimizer(parameters: list[torch.nn.Parameter], learning_rate: float):
    import bitsandbytes as bnb
    return bnb.optim.AdamW8bit(
        parameters,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )


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
    encoder_hidden_states = _batch_tensor(batch, "encoder_hidden_states").to(
        device=unet.device,
        dtype=weight_dtype,
    )
    noise = torch.randn_like(latents)
    batch_size = latents.shape[0]
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (batch_size,),
        device=latents.device,
    ).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    target = _target(
        noise_scheduler=noise_scheduler,
        latents=latents,
        noise=noise,
        timesteps=timesteps,
    )
    prediction = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states,
        return_dict=False,
    )[0]
    per_example_loss = denoising_loss_per_example(
        model_pred=prediction,
        target=target,
    )
    loss = _objective_loss(
        per_example_loss=per_example_loss,
        timesteps=timesteps,
        noise_scheduler=noise_scheduler,
        prediction_type=str(noise_scheduler.config.prediction_type),
        objective=objective,
    )
    return _LossContext(
        loss=loss,
        noise=noise.detach().cpu(),
        timesteps=timesteps.detach().cpu(),
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


def _snr_for_timesteps(
    *,
    noise_scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(
        device=timesteps.device,
        dtype=torch.float32,
    )
    alpha = torch.sqrt(alphas_cumprod[timesteps])
    sigma = torch.sqrt(1.0 - alphas_cumprod[timesteps]).clamp_min(1e-12)
    return (alpha / sigma) ** 2


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
    snr = _snr_for_timesteps(
        noise_scheduler=noise_scheduler,
        timesteps=timesteps.detach().cpu().long(),
    ).detach().float().cpu()

    window_timesteps: list[int] = []
    window_snr_values: list[float] = []
    window_record_indices: list[int] = []
    window_images: list[str] = []
    window_sources: list[str] = []
    window_latent_item_shapes: list[list[int]] = []

    for probe_batch in probe_batches:
        probe_latents = _batch_tensor(probe_batch.batch, "latents")
        probe_timesteps = probe_batch.timesteps.detach().cpu().long()
        probe_snr = _snr_for_timesteps(
            noise_scheduler=noise_scheduler,
            timesteps=probe_timesteps,
        ).detach().float().cpu()

        batch_size = int(probe_latents.shape[0])
        window_timesteps.extend(int(item) for item in probe_timesteps.tolist())
        window_snr_values.extend(float(item) for item in probe_snr.tolist())
        window_record_indices.extend(_jsonable_int_list(probe_batch.batch.get("record_indices")))
        window_images.extend(_jsonable_str_list(probe_batch.batch.get("images")))
        window_sources.extend([probe_batch.source] * batch_size)
        window_latent_item_shapes.extend(
            [[int(item) for item in probe_latents.shape[1:]]] * batch_size
        )

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


def _write_probe_summary(*, probe_log_path: Path, summary_path: Path) -> None:
    probes: list[dict[str, object]] = []
    validation_reports: list[dict[str, object]] = []
    with probe_log_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict) and "candidate_selection" in payload:
                probes.append(payload)
            if isinstance(payload, dict) and payload.get("probe_status") == "validation":
                validation_reports.append(payload)

    probe_count = len(probes)
    if probe_count == 0:
        summary = {
            "probe_count": 0,
            "adamw_selected": 0,
            "adamw_rejected": 0,
            "fallback_used": 0,
            "updates_skipped": 0,
            "mean_dual_thickness": 0.0,
            "max_dual_thickness": 0.0,
            "conflict_probe_fraction": 0.0,
            "validation_count": len(validation_reports),
            "best_checkpoint_step": _best_validation_step(validation_reports),
        }
    else:
        dual_thickness = [float(probe.get("dual_thickness", 0.0)) for probe in probes]
        conflict_count = sum(
            1
            for probe in probes
            if float(probe.get("negative_cosine_fraction", 0.0)) > 0.0
        )
        guardrails = [
            probe.get("candidate_selection", {}).get("guardrail", {})
            for probe in probes
            if isinstance(probe.get("candidate_selection"), dict)
        ]
        summary = {
            "probe_count": probe_count,
            "adamw_selected": sum(
                1
                for guardrail in guardrails
                if guardrail.get("committed_update") == "adamw_actual"
            ),
            "adamw_rejected": sum(
                1
                for probe in probes
                if not probe.get("candidate_certificates", {})
                .get("adamw_actual", {})
                .get("accepted_bottleneck", False)
            ),
            "fallback_used": sum(
                1
                for guardrail in guardrails
                if bool(guardrail.get("fallback_used", False))
            ),
            "updates_skipped": sum(
                1
                for guardrail in guardrails
                if bool(guardrail.get("skipped_update", False))
            ),
            "mean_dual_thickness": float(sum(dual_thickness) / probe_count),
            "max_dual_thickness": float(max(dual_thickness)),
            "conflict_probe_fraction": float(conflict_count / probe_count),
            "validation_count": len(validation_reports),
            "best_checkpoint_step": _best_validation_step(validation_reports),
        }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _best_validation_step(validation_reports: list[dict[str, object]]) -> int | None:
    improved = [
        report
        for report in validation_reports
        if bool(report.get("best_checkpoint_improved", False))
    ]
    if not improved:
        return None
    return int(improved[-1]["step"])


def _evaluate_validation_skeleton(
    *,
    skeleton: _ValidationSkeleton,
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    step: int,
) -> _ValidationReport:
    if not skeleton.items:
        raise LorakitError("Validation skeleton has no items")
    losses_by_bucket: dict[str, list[float]] = {}
    for item in skeleton.items:
        loss = _validation_item_loss(
            item=item,
            unet=unet,
            noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
        )
        losses_by_bucket.setdefault(item.snr_bucket, []).append(loss)
    bucket_means = {
        bucket: float(sum(losses) / len(losses))
        for bucket, losses in losses_by_bucket.items()
    }
    all_losses = [
        loss
        for losses in losses_by_bucket.values()
        for loss in losses
    ]
    return _ValidationReport(
        step=int(step),
        loss_mean=float(sum(all_losses) / len(all_losses)),
        loss_max_snr_bucket=float(max(bucket_means.values())),
        loss_by_snr_bucket=bucket_means,
        item_count=len(all_losses),
    )


@torch.no_grad()
def _validation_item_loss(
    *,
    item: _ValidationItem,
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
) -> float:
    latent = _load_tensor(item.record.latent_path, expected_shape=item.record.latent_shape)
    hidden = _load_tensor(
        item.record.encoder_hidden_state_path,
        expected_shape=item.record.encoder_hidden_state_shape,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(item.noise_seed)
    noise = torch.randn(
        item.record.latent_shape,
        generator=generator,
        dtype=torch.float32,
    )
    batch = {
        "latents": latent.unsqueeze(0),
        "encoder_hidden_states": hidden.unsqueeze(0),
    }
    loss = _fixed_subset_context_loss(
        batch=batch,
        mask=torch.tensor([True], device=unet.device),
        unet=unet,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
        noise=noise.unsqueeze(0),
        timesteps=torch.tensor([item.timestep], dtype=torch.long),
    )
    return float(loss.detach().float().cpu().item())


def _select_best_checkpoint(
    *,
    current: _BestCheckpoint,
    report: _ValidationReport,
) -> tuple[bool, _BestCheckpoint]:
    if current.loss_max_snr_bucket is None or current.loss_mean is None or current.step is None:
        return True, _BestCheckpoint(
            path=current.path,
            step=report.step,
            loss_max_snr_bucket=report.loss_max_snr_bucket,
            loss_mean=report.loss_mean,
        )
    candidate_key = (
        report.loss_max_snr_bucket,
        report.loss_mean,
        report.step,
    )
    current_key = (
        current.loss_max_snr_bucket,
        current.loss_mean,
        current.step,
    )
    if candidate_key < current_key:
        return True, _BestCheckpoint(
            path=current.path,
            step=report.step,
            loss_max_snr_bucket=report.loss_max_snr_bucket,
            loss_mean=report.loss_mean,
        )
    return False, current


def _write_validation_report_if_main(
    *,
    probe_log_path: Path,
    report: _ValidationReport,
    best_checkpoint: _BestCheckpoint,
    improved: bool,
    should_log: bool,
) -> None:
    if not should_log:
        return
    payload = {
        "step": int(report.step),
        "probe_status": "validation",
        "validation_loss_mean": float(report.loss_mean),
        "validation_loss_max_snr_bucket": float(report.loss_max_snr_bucket),
        "validation_loss_by_snr_bucket": {
            bucket: float(loss)
            for bucket, loss in sorted(report.loss_by_snr_bucket.items())
        },
        "validation_item_count": int(report.item_count),
        "best_checkpoint_improved": bool(improved),
        "best_checkpoint_step": best_checkpoint.step,
        "best_checkpoint_path": str(best_checkpoint.path),
    }
    with probe_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


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


def _challenge_quality_policies(
    *,
    current_policy: _TrainingPolicy,
    batch: dict[str, object],
    unet: UNet2DConditionModel,
    optimizer,
    validation_skeleton: _ValidationSkeleton,
    baseline_report: _ValidationReport,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    learning_rate: float,
) -> tuple[_TrainingPolicy, dict[str, object]]:
    parameter_snapshot = _snapshot_trainable_parameters(certified_trainable_parameters(unet))
    parameters = certified_trainable_parameters(unet)
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
    objective_reports = {}
    best_objective = current_policy.objective
    best_objective_report = baseline_report
    for objective in objective_candidates:
        report = _virtual_policy_step_validation_report(
            batch=batch,
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
        )
        objective_reports[_objective_label(objective)] = _validation_delta_log(
            baseline=baseline_report,
            candidate=report,
            gamma=objective.gamma,
        )
        if _validation_score_improves(
            baseline=best_objective_report,
            candidate=report,
        ):
            best_objective = objective
            best_objective_report = report

    ratio_candidates = _lora_plus_candidate_ratios(unet)
    ratio_reports = {}
    best_ratio = current_policy.lora_plus_ratio
    best_ratio_report = best_objective_report
    scale_a, scale_b = _lora_relative_gradient_scales(unet)
    for ratio in ratio_candidates:
        report = _virtual_policy_step_validation_report(
            batch=batch,
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
        )
        ratio_reports[_ratio_label(ratio)] = _validation_delta_log(
            baseline=baseline_report,
            candidate=report,
        )
        if _validation_score_improves(
            baseline=best_ratio_report,
            candidate=report,
        ):
            best_ratio = ratio
            best_ratio_report = report

    _restore_trainable_parameters(parameters, parameter_snapshot)
    _restore_trainable_gradients(parameters, gradient_snapshot)
    _restore_optimizer_state(optimizer, optimizer_snapshot)

    next_policy = _TrainingPolicy(
        objective=best_objective,
        lora_plus_ratio=best_ratio,
    )
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
        "rank_probe": {
            "trigger": "not_run_before_capacity_probe_milestone",
            "rank_growth_accepted": False,
        },
    }


def _virtual_policy_step_validation_report(
    *,
    batch: dict[str, object],
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
) -> _ValidationReport:
    _restore_trainable_parameters(parameters, parameter_snapshot)
    _restore_trainable_gradients(parameters, gradient_snapshot)
    _restore_optimizer_state(optimizer, optimizer_snapshot)
    optimizer.zero_grad(set_to_none=True)
    loss_context = _loss_context(
        batch=batch,
        unet=unet,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
        objective=objective,
    )
    loss_context.loss.backward()
    _apply_lora_plus_gradient_ratio(unet=unet, ratio=lora_plus_ratio)
    optimizer.step()
    with torch.no_grad():
        report = _evaluate_validation_skeleton(
            skeleton=validation_skeleton,
            unet=unet,
            noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
            step=-1,
        )
    return report


def _objective_candidates(
    *,
    current_policy: _TrainingPolicy,
    baseline_report: _ValidationReport,
    validation_skeleton: _ValidationSkeleton,
    noise_scheduler: DDPMScheduler,
) -> list[_ObjectivePolicy]:
    candidates = [_ObjectivePolicy(name="base_mse")]
    gammas = [
        _validation_bucket_gamma(
            validation_skeleton=validation_skeleton,
            bucket="mid",
        ),
        5.0,
        _validation_bucket_gamma(
            validation_skeleton=validation_skeleton,
            bucket=max(baseline_report.loss_by_snr_bucket, key=baseline_report.loss_by_snr_bucket.get),
        ),
    ]
    if current_policy.objective.name == "minsnr" and current_policy.objective.gamma is not None:
        gammas.append(current_policy.objective.gamma)
    seen = set()
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


def _validation_delta_log(
    *,
    baseline: _ValidationReport,
    candidate: _ValidationReport,
    gamma: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "validation_loss_max_snr_bucket_delta": float(candidate.loss_max_snr_bucket - baseline.loss_max_snr_bucket),
        "validation_loss_mean_delta": float(candidate.loss_mean - baseline.loss_mean),
    }
    if gamma is not None:
        payload["gamma"] = float(gamma)
    return payload


def _validation_score_improves(*, baseline: _ValidationReport, candidate: _ValidationReport) -> bool:
    tolerance = _validation_score_tolerance(baseline)
    if candidate.loss_max_snr_bucket < baseline.loss_max_snr_bucket - tolerance:
        return True
    if candidate.loss_max_snr_bucket > baseline.loss_max_snr_bucket + tolerance:
        return False
    return candidate.loss_mean < baseline.loss_mean - tolerance


def _validation_score_tolerance(report: _ValidationReport) -> float:
    return max(
        VALIDATION_SCORE_ABSOLUTE_TOLERANCE,
        VALIDATION_SCORE_RELATIVE_TOLERANCE * max(report.loss_by_snr_bucket.values()),
    )


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


def _partition_lora_parameters(
    unet: UNet2DConditionModel,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
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


def _apply_lora_plus_gradient_ratio(*, unet: UNet2DConditionModel, ratio: float) -> None:
    if abs(float(ratio) - 1.0) <= 1e-12:
        return
    _lora_a, lora_b = _partition_lora_parameters(unet)
    for parameter in lora_b:
        if parameter.grad is not None:
            parameter.grad.mul_(float(ratio))


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

    (
        context_ids,
        context_counts,
        context_losses_tensor,
        gradients,
    ) = _collect_probe_context_gradients(
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
    (
        candidate_certificates,
        candidate_selection,
        guardrail_decision,
    ) = _candidate_step_certificates_for_probe_window(
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
    if not candidates:
        return probe_batches

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
    context_losses = torch.stack(
        [
            loss_sums[context_id] / float(counts[context_id])
            for context_id in context_ids
        ]
    )
    gradients = [
        gradient_sums[context_id] / float(counts[context_id])
        for context_id in context_ids
    ]

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
        "mean_context_sgd_proxy": weighted_gradient(
            gradients=gradients,
            weights=mean_context_weights,
        ),
        "mean_example_sgd_proxy": weighted_gradient(
            gradients=gradients,
            weights=count_weights,
        ),
        "mgda_sgd_proxy": weighted_gradient(
            gradients=gradients,
            weights=mgda_weights,
        ),
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
        else:
            results["adamw_actual"] = {
                "available": False,
                "reason": "optimizer_state_snapshot_unavailable",
            }

        for name, flat_gradient in candidate_gradients.items():
            _restore_trainable_parameters(parameters, parameter_snapshot)
            _restore_trainable_gradients(parameters, gradient_snapshot)
            if optimizer_snapshot is not None:
                _restore_optimizer_state(optimizer, optimizer_snapshot)
            update_norms[name] = float(torch.linalg.vector_norm(flat_gradient.detach().float().cpu()).item())
            _apply_flat_gradient_step(
                parameters=parameters,
                flat_gradient=flat_gradient,
                step_size=step_size,
            )
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
    finally:
        _restore_trainable_parameters(parameters, parameter_snapshot)
        _restore_trainable_gradients(parameters, gradient_snapshot)
        if optimizer_snapshot is not None:
            _restore_optimizer_state(optimizer, optimizer_snapshot)

    selection = select_preferred_candidate(
        certificate_objects,
        update_norms=update_norms,
    )
    guardrail_decision = _commit_certified_guardrail_update(
        unet=unet,
        optimizer=optimizer,
        parameters=parameters,
        parameter_snapshot=parameter_snapshot,
        gradient_snapshot=gradient_snapshot,
        optimizer_snapshot=optimizer_snapshot,
        probe_batches=probe_batches,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
        context_ids=context_ids,
        old_context_losses=old_context_losses,
        candidate_gradients=candidate_gradients,
        candidate_certificates=certificate_objects,
        step_size=step_size,
    )
    selection["committed_update"] = guardrail_decision.committed_update
    selection["selection_is_instrumentation_only"] = False
    selection["update_norms"] = {
        name: float(value)
        for name, value in update_norms.items()
    }
    selection["guardrail"] = {
        "enabled": True,
        "probe_step": True,
        "initial_candidate": "adamw_actual",
        "committed_update": guardrail_decision.committed_update,
        "backtracks": int(guardrail_decision.backtracks),
        "fallback_used": bool(guardrail_decision.fallback_used),
        "skipped_update": bool(guardrail_decision.skipped_update),
    }

    return results, selection, guardrail_decision


def _commit_certified_guardrail_update(
    *,
    unet: UNet2DConditionModel,
    optimizer,
    parameters: list[torch.nn.Parameter],
    parameter_snapshot: list[torch.Tensor],
    gradient_snapshot: list[torch.Tensor | None],
    optimizer_snapshot,
    probe_batches: list[_ProbeBatch],
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    context_ids: tuple[int, ...],
    old_context_losses: torch.Tensor,
    candidate_gradients: dict[str, torch.Tensor],
    candidate_certificates: dict[str, StepCertificate],
    step_size: float,
) -> _GuardrailDecision:
    adamw_certificate = candidate_certificates.get("adamw_actual")
    if optimizer_snapshot is not None and adamw_certificate is not None and adamw_certificate.accepted_bottleneck:
        _restore_trainable_parameters(parameters, parameter_snapshot)
        _restore_trainable_gradients(parameters, gradient_snapshot)
        _restore_optimizer_state(optimizer, optimizer_snapshot)
        optimizer.step()
        return _GuardrailDecision(
            handled_update=True,
            committed_update="adamw_actual",
            backtracks=0,
            fallback_used=False,
            skipped_update=False,
        )

    if optimizer_snapshot is not None:
        for backtrack_index, factor in enumerate(CERTIFIED_BACKTRACK_FACTORS[1:], start=1):
            certificate = _try_backtracked_adamw_update(
                unet=unet,
                optimizer=optimizer,
                parameters=parameters,
                parameter_snapshot=parameter_snapshot,
                gradient_snapshot=gradient_snapshot,
                optimizer_snapshot=optimizer_snapshot,
                probe_batches=probe_batches,
                noise_scheduler=noise_scheduler,
                weight_dtype=weight_dtype,
                context_ids=context_ids,
                old_context_losses=old_context_losses,
                step_size=step_size * float(factor),
                factor=float(factor),
                backtracks=backtrack_index,
            )
            if certificate.accepted_bottleneck:
                return _GuardrailDecision(
                    handled_update=True,
                    committed_update="adamw_actual_backtracked",
                    backtracks=backtrack_index,
                    fallback_used=False,
                    skipped_update=False,
                )

    for name in ("mgda_sgd_proxy", "mean_context_sgd_proxy"):
        certificate = candidate_certificates.get(name)
        flat_gradient = candidate_gradients.get(name)
        if certificate is None or flat_gradient is None or not certificate.accepted_bottleneck:
            continue
        _restore_trainable_parameters(parameters, parameter_snapshot)
        _restore_trainable_gradients(parameters, gradient_snapshot)
        if optimizer_snapshot is not None:
            _restore_optimizer_state(optimizer, optimizer_snapshot)
        _apply_flat_gradient_step(
            parameters=parameters,
            flat_gradient=flat_gradient,
            step_size=step_size,
        )
        return _GuardrailDecision(
            handled_update=True,
            committed_update=name,
            backtracks=0,
            fallback_used=True,
            skipped_update=False,
        )

    _restore_trainable_parameters(parameters, parameter_snapshot)
    _restore_trainable_gradients(parameters, gradient_snapshot)
    if optimizer_snapshot is not None:
        _restore_optimizer_state(optimizer, optimizer_snapshot)
    return _GuardrailDecision(
        handled_update=True,
        committed_update="skip_update",
        backtracks=0,
        fallback_used=False,
        skipped_update=True,
    )


def _try_backtracked_adamw_update(
    *,
    unet: UNet2DConditionModel,
    optimizer,
    parameters: list[torch.nn.Parameter],
    parameter_snapshot: list[torch.Tensor],
    gradient_snapshot: list[torch.Tensor | None],
    optimizer_snapshot,
    probe_batches: list[_ProbeBatch],
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
    context_ids: tuple[int, ...],
    old_context_losses: torch.Tensor,
    step_size: float,
    factor: float,
    backtracks: int,
) -> StepCertificate:
    _restore_trainable_parameters(parameters, parameter_snapshot)
    _restore_trainable_gradients(parameters, gradient_snapshot)
    _restore_optimizer_state(optimizer, optimizer_snapshot)
    before = _snapshot_trainable_parameters(parameters)
    optimizer.step()
    _scale_trainable_update(
        parameters=parameters,
        before=before,
        factor=factor,
    )
    with torch.no_grad():
        new_context_losses = _evaluate_probe_context_losses(
            probe_batches=probe_batches,
            unet=unet,
            noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
            context_ids=context_ids,
        )
    return certify_losses(
        old_context_losses=old_context_losses,
        new_context_losses=new_context_losses,
        backtracks=backtracks,
        step_size=step_size,
        context_ids=context_ids,
    )


def _save_named_lora_weights(
    *,
    unet: UNet2DConditionModel,
    output_dir: Path,
    filename: str,
) -> Path:
    temporary_dir = output_dir / f".{filename}.tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    lora_layers = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(unet)
    )
    StableDiffusionPipeline.save_lora_weights(
        save_directory=temporary_dir,
        unet_lora_layers=lora_layers,
        safe_serialization=True,
    )
    source = temporary_dir / LORA_WEIGHTS_NAME
    destination = output_dir / filename
    shutil.copy2(source, destination)
    shutil.rmtree(temporary_dir)
    return destination


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


def _flatten_trainable_parameters(
    parameters: list[torch.nn.Parameter],
) -> torch.Tensor:
    pieces = [parameter.detach().float().reshape(-1).cpu() for parameter in parameters]
    if not pieces:
        raise LorakitError("No trainable parameters found for candidate evaluation")
    return torch.cat(pieces, dim=0)


def _snapshot_trainable_gradients(
    parameters: list[torch.nn.Parameter],
) -> list[torch.Tensor | None]:
    return [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]


@torch.no_grad()
def _restore_trainable_gradients(
    parameters: list[torch.nn.Parameter],
    snapshot: list[torch.Tensor | None],
) -> None:
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


def _snapshot_trainable_parameters(
    parameters: list[torch.nn.Parameter],
) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


@torch.no_grad()
def _restore_trainable_parameters(
    parameters: list[torch.nn.Parameter],
    snapshot: list[torch.Tensor],
) -> None:
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
    encoder_hidden_states = _batch_tensor(batch, "encoder_hidden_states").to(
        device=unet.device,
        dtype=weight_dtype,
    )[mask]
    local_noise = noise.to(device=unet.device, dtype=weight_dtype)[mask]
    local_timesteps = timesteps.to(device=unet.device).long()[mask]
    noisy_latents = noise_scheduler.add_noise(latents, local_noise, local_timesteps)
    target = _target(
        noise_scheduler=noise_scheduler,
        latents=latents,
        noise=local_noise,
        timesteps=local_timesteps,
    )
    prediction = unet(
        noisy_latents,
        local_timesteps,
        encoder_hidden_states,
        return_dict=False,
    )[0]
    return denoising_loss_per_example(
        model_pred=prediction,
        target=target,
    ).mean()


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
