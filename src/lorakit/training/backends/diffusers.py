"""Diffusers LoRA training backend."""

import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers

from lorakit.errors import LorakitError
from lorakit.manifest import MANIFEST_NAME, read_manifest
from lorakit.training.backends.types import BackendResult, BackendSpec


LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]
LORA_WEIGHTS_NAME = "pytorch_lora_weights.safetensors"
MAX_GRAD_NORM = 1.0
LR_WARMUP_STEPS = 0
VAE_DOWNSAMPLE_FACTOR: Final = 8
CACHE_ENCODING_BATCH_SIZE: Final = 4
CACHE_DIR_NAME: Final = "tensor-cache"
CACHE_MANIFEST_NAME: Final = "manifest.json"
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
    dataset = _DiskCachedLatentDataset(
        records=cache_records,
    )
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=spec.batch_size,
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
    while global_step < spec.steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                loss = _loss(
                    batch=batch,
                    unet=unet,
                    noise_scheduler=noise_scheduler,
                    weight_dtype=weight_dtype,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [parameter for parameter in unet.parameters() if parameter.requires_grad],
                        MAX_GRAD_NORM,
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
    return BackendResult(model_path=model_path)


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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self._records[index]
        return {
            "latents": _load_tensor(record.latent_path, expected_shape=record.latent_shape),
            "encoder_hidden_states": _load_tensor(
                record.encoder_hidden_state_path,
                expected_shape=record.encoder_hidden_state_shape,
            ),
        }


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
    expected_size: tuple[int, int] | None = None
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
        current_size = (width, height)
        if expected_size is None:
            expected_size = current_size
        elif current_size != expected_size:
            raise LorakitError(
                "Prepared images must share one tensor shape for batched training: "
                f"expected {expected_size[0]}x{expected_size[1]}, got {width}x{height} at {image_path}"
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
    cached: list[Path] = []
    for batch_start in tqdm(
        range(0, len(rows), CACHE_ENCODING_BATCH_SIZE),
        desc="Encoding captions",
    ):
        batch_rows = rows[batch_start : batch_start + CACHE_ENCODING_BATCH_SIZE]
        tokens = tokenizer(
            [row.caption for row in batch_rows],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        hidden = text_encoder(tokens, return_dict=False)[0]
        for offset, tensor in enumerate(hidden):
            path = text_cache_dir / f"{batch_start + offset:08d}.pt"
            _save_tensor(path, tensor.detach().to(dtype=dtype).cpu())
            cached.append(path)
    return cached


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
    cached: list[Path] = []
    for batch_start in tqdm(
        range(0, len(rows), CACHE_ENCODING_BATCH_SIZE),
        desc="Encoding latents",
    ):
        batch_rows = rows[batch_start : batch_start + CACHE_ENCODING_BATCH_SIZE]
        pixel_batches: list[torch.Tensor] = []
        for row in batch_rows:
            with Image.open(row.image_path) as image:
                pixel_batches.append(transform(image.convert("RGB")))
        pixel_values = torch.stack(pixel_batches).to(device=device, dtype=dtype)
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        for offset, tensor in enumerate(latents):
            path = latent_cache_dir / f"{batch_start + offset:08d}.pt"
            _save_tensor(path, tensor.detach().to(dtype=dtype).cpu())
            cached.append(path)
    return cached


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
        if record.latent_shape != first.latent_shape:
            raise LorakitError(
                "Latent cache tensor shape mismatch: "
                f"expected {first.latent_shape}, got {record.latent_shape} for {record.image}"
            )
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
        "latent_shape": list(records[0].latent_shape),
        "encoder_hidden_state_shape": list(records[0].encoder_hidden_state_shape),
        "records": [
            {
                "image": record.image,
                "image_sha256": record.image_sha256,
                "caption_sha256": record.caption_sha256,
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


def _collate_cached(examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "latents": torch.stack([example["latents"] for example in examples]).to(
            memory_format=torch.contiguous_format
        ),
        "encoder_hidden_states": torch.stack(
            [example["encoder_hidden_states"] for example in examples]
        ).to(memory_format=torch.contiguous_format),
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


def _loss(
    *,
    batch: dict[str, torch.Tensor],
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    latents = batch["latents"].to(device=unet.device, dtype=weight_dtype)
    encoder_hidden_states = batch["encoder_hidden_states"].to(
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
    return F.mse_loss(prediction.float(), target.float(), reduction="mean")


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
