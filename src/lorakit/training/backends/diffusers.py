"""Diffusers LoRA training backend."""

import copy
import gc
import hashlib
import json
import random
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
MAX_GRAD_NORM = 1.0
LR_WARMUP_STEPS = 0
VAE_DOWNSAMPLE_FACTOR: Final = 8
CACHE_ENCODING_MAX_BATCH_SIZE: Final = 4
CACHE_ENCODING_MIN_BATCH_SIZE: Final = 1
CACHE_DIR_NAME: Final = "tensor-cache"
CACHE_MANIFEST_NAME: Final = "manifest.json"
PROBE_LOG_NAME = "context-probes.jsonl"
PROBE_INITIAL_STEPS: Final = 3
PROBE_EVERY_STEPS: Final = 25
PROBE_MIN_FREE_CUDA_BYTES: Final = 500_000_000
PROBE_WINDOW_TARGET_ITEMS: Final = 4
PROBE_VERSION: Final = 4
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
    _write_cache_manifest(cache_dir / CACHE_MANIFEST_NAME, cache_records)
    dataset = _DiskCachedLatentDataset(records=cache_records)
    dataloader = DataLoader(
        dataset,
        batch_sampler=_LatentShapeBatchSampler(
            records=cache_records,
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
    while global_step < spec.steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                loss_context = _loss_context(
                    batch=batch,
                    unet=unet,
                    noise_scheduler=noise_scheduler,
                    weight_dtype=weight_dtype,
                )
                loss = loss_context.loss
                accelerator.backward(loss)
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
                        _write_streaming_context_probe_if_main(
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
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=float(loss.detach().item()))
                if global_step >= spec.steps:
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        lora_layers = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_unet)
        )
        StableDiffusionPipeline.save_lora_weights(
            save_directory=spec.output_dir,
            unet_lora_layers=lora_layers,
            safe_serialization=True,
        )
    accelerator.end_training()

    model_path = spec.output_dir / LORA_WEIGHTS_NAME
    if not model_path.exists():
        raise LorakitError(f"Diffusers backend did not write LoRA weights: {model_path}")
    artifact_paths = (probe_log_path,) if probe_log_path.exists() else ()
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
        latents = vae.encode(pixel_values).latent_dist.sample()
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


def _write_cache_manifest(path: Path, records: list[_CacheRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "count": len(records),
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
    return _LossContext(
        loss=per_example_loss.mean(),
        noise=noise.detach().cpu(),
        timesteps=timesteps.detach().cpu(),
    )


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
) -> None:
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

    if not should_log or not gradients:
        return

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
    candidate_certificates, candidate_selection = _candidate_step_certificates_for_probe_window(
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

    payload = probe_to_log_dict(probe, step=step, extra=extra)
    probe_log_path.parent.mkdir(parents=True, exist_ok=True)
    with probe_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


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
) -> tuple[dict[str, object], dict[str, object]]:
    parameter_snapshot = _snapshot_trainable_parameters(parameters)
    gradient_snapshot = _snapshot_trainable_gradients(parameters)
    optimizer_snapshot = _snapshot_optimizer_state(optimizer)

    certificate_objects = {}
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
    selection["committed_update"] = "adamw_actual"
    selection["selection_is_instrumentation_only"] = True
    selection["update_norms"] = {
        name: float(value)
        for name, value in update_norms.items()
    }

    return results, selection

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
    except Exception:
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
