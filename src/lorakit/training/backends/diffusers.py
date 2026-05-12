"""Diffusers LoRA training backend."""

import gc
from pathlib import Path

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
    encoder_hidden_states = _cache_encoder_hidden_states(
        rows=rows,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=accelerator.device,
        dtype=weight_dtype,
    )
    latents = _cache_latents(
        rows=rows,
        prepared_dir=spec.prepared_dir,
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

    dataset = _CachedLatentDataset(
        latents=latents,
        encoder_hidden_states=encoder_hidden_states,
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


class _CachedLatentDataset(Dataset):
    def __init__(
        self,
        *,
        latents: list[torch.Tensor],
        encoder_hidden_states: list[torch.Tensor],
    ):
        if len(latents) != len(encoder_hidden_states):
            raise LorakitError("Latent cache and text embedding cache length mismatch")
        self._latents = latents
        self._encoder_hidden_states = encoder_hidden_states

    def __len__(self) -> int:
        return len(self._latents)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "latents": self._latents[index],
            "encoder_hidden_states": self._encoder_hidden_states[index],
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


def _load_rows(prepared_dir: Path) -> list[dict[str, object]]:
    rows = read_manifest(prepared_dir / MANIFEST_NAME)
    if not rows:
        raise LorakitError(f"Prepared dataset has no training rows: {prepared_dir}")
    return rows


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
    rows: list[dict[str, object]],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    text_encoder.requires_grad_(False)
    text_encoder.to(device=device, dtype=dtype)
    text_encoder.eval()
    cached: list[torch.Tensor] = []
    for row in tqdm(rows, desc="Encoding captions"):
        tokens = tokenizer(
            str(row["caption"]),
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        hidden = text_encoder(tokens, return_dict=False)[0]
        cached.append(hidden[0].detach().to(dtype=dtype).cpu())
    return cached


@torch.no_grad()
def _cache_latents(
    *,
    rows: list[dict[str, object]],
    prepared_dir: Path,
    vae: AutoencoderKL,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    vae.requires_grad_(False)
    vae.to(device=device, dtype=dtype)
    vae.eval()
    cached: list[torch.Tensor] = []
    for row in tqdm(rows, desc="Encoding latents"):
        image_path = (prepared_dir / str(row["image"])).resolve()
        with Image.open(image_path) as image:
            pixel_values = transform(image.convert("RGB")).unsqueeze(0)
            pixel_values = pixel_values.to(device=device, dtype=dtype)
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        cached.append(latents[0].detach().to(dtype=dtype).cpu())
    return cached


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
