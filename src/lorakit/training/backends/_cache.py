"""Disk-backed tensor cache helpers for the Diffusers training backend."""

from __future__ import annotations

import gc
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

import torch
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import BatchSampler, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from lorakit.errors import LorakitError
from lorakit.manifest import MANIFEST_NAME, read_manifest


CACHE_DIR_NAME: Final = "tensor-cache"
CACHE_MANIFEST_NAME: Final = "manifest.json"
CACHE_ENCODING_MAX_BATCH_SIZE: Final = 4
CACHE_ENCODING_MIN_BATCH_SIZE: Final = 1
HASH_CHUNK_SIZE_BYTES: Final = 1024 * 1024
LATENT_CACHE_MODE: Final = "posterior_mode"
VAE_DOWNSAMPLE_FACTOR: Final = 8


@dataclass(frozen=True)
class TrainingRow:
    image: str
    caption: str
    image_path: Path
    image_sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class CacheRecord:
    image: str
    caption_sha256: str
    image_sha256: str
    latent_path: Path
    encoder_hidden_state_path: Path
    latent_shape: tuple[int, ...]
    encoder_hidden_state_shape: tuple[int, ...]


class DiskCachedLatentDataset(Dataset):
    def __init__(self, *, records: list[CacheRecord]):
        if not records:
            raise LorakitError("Tensor cache has no records")
        assert_cache_records_are_consistent(records)
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self._records[index]
        return {
            "latents": load_tensor(record.latent_path, expected_shape=record.latent_shape),
            "encoder_hidden_states": load_tensor(
                record.encoder_hidden_state_path,
                expected_shape=record.encoder_hidden_state_shape,
            ),
            "record_index": index,
            "image": record.image,
            "latent_shape": record.latent_shape,
            "encoder_hidden_state_shape": record.encoder_hidden_state_shape,
        }


class LatentShapeBatchSampler(BatchSampler):
    def __init__(self, *, records: list[CacheRecord], batch_size: int):
        if batch_size <= 0:
            raise LorakitError("Batch size must be greater than zero")
        buckets = latent_shape_buckets(records)
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


def load_rows(prepared_dir: Path) -> list[TrainingRow]:
    rows = read_manifest(prepared_dir / MANIFEST_NAME)
    if not rows:
        raise LorakitError(f"Prepared dataset has no training rows: {prepared_dir}")

    loaded_rows: list[TrainingRow] = []
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
            TrainingRow(
                image=image,
                caption=caption,
                image_path=image_path,
                image_sha256=file_sha256(image_path),
                width=width,
                height=height,
            )
        )
    return loaded_rows


def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "cuda out of memory" in str(exc).lower()


def cleanup_after_cuda_oom() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def adaptive_cuda_batches(
    *,
    indices: list[int],
    initial_batch_size: int,
    encode_batch: Callable[[list[int]], None],
) -> None:
    batch_size = max(CACHE_ENCODING_MIN_BATCH_SIZE, int(initial_batch_size))
    cursor = 0
    while cursor < len(indices):
        current = indices[cursor : cursor + batch_size]
        try:
            encode_batch(current)
            cursor += len(current)
        except RuntimeError as exc:
            if not is_cuda_oom(exc):
                raise
            cleanup_after_cuda_oom()
            if batch_size <= CACHE_ENCODING_MIN_BATCH_SIZE:
                raise LorakitError(
                    "CUDA out of memory while encoding a single cache item. "
                    "Try closing other GPU processes or preparing at a smaller resolution."
                ) from exc
            batch_size = max(CACHE_ENCODING_MIN_BATCH_SIZE, batch_size // 2)


@torch.no_grad()
def cache_encoder_hidden_states(
    *,
    rows: list[TrainingRow],
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
            save_tensor(path, tensor.detach().to(dtype=dtype).cpu())
            cached[index] = path
        progress.update(len(batch_indices))
        del tokens
        del hidden

    try:
        adaptive_cuda_batches(
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
def cache_latents(
    *,
    rows: list[TrainingRow],
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
    shape_buckets = pixel_shape_buckets(rows)
    progress = tqdm(total=len(rows), desc="Encoding latents")

    def encode_batch(batch_indices: list[int]) -> None:
        pixel_batches: list[torch.Tensor] = []
        for index in batch_indices:
            with Image.open(rows[index].image_path) as image:
                pixel_batches.append(transform(image.convert("RGB")))
        pixel_values = torch.stack(pixel_batches).to(device=device, dtype=dtype)
        latent_distribution = vae.encode(pixel_values).latent_dist
        latents = deterministic_latents(latent_distribution)
        latents = latents * vae.config.scaling_factor
        for index, tensor in zip(batch_indices, latents, strict=True):
            path = latent_cache_dir / f"{index:08d}.pt"
            save_tensor(path, tensor.detach().to(dtype=dtype).cpu())
            cached[index] = path
        progress.update(len(batch_indices))
        del pixel_batches
        del pixel_values
        del latents

    try:
        for indices in shape_buckets.values():
            adaptive_cuda_batches(
                indices=indices,
                initial_batch_size=CACHE_ENCODING_MAX_BATCH_SIZE,
                encode_batch=encode_batch,
            )
    finally:
        progress.close()

    if any(path is None for path in cached):
        raise LorakitError("Latent cache did not encode every row")
    return [path for path in cached if path is not None]


def deterministic_latents(latent_distribution) -> torch.Tensor:
    if hasattr(latent_distribution, "mode"):
        mode = latent_distribution.mode()
        if isinstance(mode, torch.Tensor):
            return mode
    mean = getattr(latent_distribution, "mean", None)
    if isinstance(mean, torch.Tensor):
        return mean
    raise LorakitError("VAE latent distribution does not expose mode() or mean")


def pixel_shape_buckets(rows: list[TrainingRow]) -> dict[tuple[int, int], list[int]]:
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(rows):
        buckets.setdefault((row.width, row.height), []).append(index)
    return buckets


def latent_shape_buckets(records: list[CacheRecord]) -> dict[tuple[int, ...], list[int]]:
    buckets: dict[tuple[int, ...], list[int]] = {}
    for index, record in enumerate(records):
        buckets.setdefault(record.latent_shape, []).append(index)
    return buckets


def cache_records(
    *,
    rows: list[TrainingRow],
    latent_paths: list[Path],
    encoder_hidden_state_paths: list[Path],
) -> list[CacheRecord]:
    if len(rows) != len(latent_paths):
        raise LorakitError("Latent cache length does not match prepared manifest length")
    if len(rows) != len(encoder_hidden_state_paths):
        raise LorakitError("Text embedding cache length does not match prepared manifest length")

    records: list[CacheRecord] = []
    for row, latent_path, hidden_path in zip(
        rows, latent_paths, encoder_hidden_state_paths, strict=True
    ):
        latent = load_tensor(latent_path)
        hidden = load_tensor(hidden_path)
        records.append(
            CacheRecord(
                image=row.image,
                caption_sha256=text_sha256(row.caption),
                image_sha256=row.image_sha256,
                latent_path=latent_path,
                encoder_hidden_state_path=hidden_path,
                latent_shape=tuple(latent.shape),
                encoder_hidden_state_shape=tuple(hidden.shape),
            )
        )
    assert_cache_records_are_consistent(records)
    return records


def write_cache_manifest(
    path: Path,
    records: list[CacheRecord],
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
        "latent_shapes": [
            list(shape)
            for shape in sorted({record.latent_shape for record in records})
        ],
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


def assert_cache_records_are_consistent(records: list[CacheRecord]) -> None:
    if not records:
        raise LorakitError("Tensor cache has no records")
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


def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def load_tensor(path: Path, expected_shape: tuple[int, ...] | None = None) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise LorakitError(f"Cached tensor is not a torch.Tensor: {path}")
    if expected_shape is not None and tuple(tensor.shape) != expected_shape:
        raise LorakitError(
            "Cached tensor shape changed for "
            f"{path}: expected {expected_shape}, got {tuple(tensor.shape)}"
        )
    return tensor


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_SIZE_BYTES)
            if chunk == b"":
                break
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def collate_cached(examples: list[dict[str, object]]) -> dict[str, object]:
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
