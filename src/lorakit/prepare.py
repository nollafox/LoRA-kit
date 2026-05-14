"""Build training-ready prepared datasets."""

import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageOps
from tqdm.auto import tqdm

import lorakit.candidates as candidates
import lorakit.datasets as datasets
from lorakit.errors import InvalidPrepareConfig, LorakitError
from lorakit.manifest import MANIFEST_NAME, build_manifest_entry, write_json, write_manifest
from lorakit.paths import Paths
from lorakit.tools.watermark import WatermarkRemover, build_watermark_remover
from lorakit.types import PrepareConfig


FORMAT_EXTENSIONS = {
    "png": ".png",
    "jpg": ".jpg",
    "webp": ".webp",
}


def prepare(paths: Paths, dataset: str, config: PrepareConfig) -> Path:
    _validate_config(config)
    dataset_dir = paths.staged_for(dataset)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise LorakitError(f"Dataset not found: {dataset}")

    staged_images = sorted(path for path in dataset_dir.iterdir() if candidates.is_image(path))
    prepared_dir = paths.prepared_for(dataset)
    if prepared_dir.exists():
        shutil.rmtree(prepared_dir)
    images_dir = prepared_dir / "images"
    images_dir.mkdir(parents=True)
    watermark_remover = (
        build_watermark_remover(
            models_dir=paths.models,
            cache_dir=paths.huggingface_cache,
            allow_download=False,
        )
        if config.remove_watermarks and len(staged_images) > 0
        else None
    )

    rows: list[dict[str, object]] = []
    target_names: set[str] = set()
    for source in tqdm(staged_images, desc="Preparing", unit="image"):
        metadata_path = datasets.require_metadata(paths, dataset, source.stem)
        target = images_dir / _prepared_filename(source, config)
        if target.name in target_names:
            raise LorakitError(f"Prepared image filename collision: {target.name}")
        target_names.add(target.name)
        _prepare_image(source, target, config, watermark_remover)
        rows.append(
            build_manifest_entry(
                image_path=target,
                metadata_path=metadata_path,
                prepared_root=prepared_dir,
                trigger=config.trigger,
                prompt_type=config.prompt_type,
            )
        )

    write_manifest(prepared_dir / MANIFEST_NAME, rows)
    write_json(prepared_dir / "config.json", asdict(config))
    return prepared_dir


def _validate_config(config: PrepareConfig) -> None:
    if config.mode not in {"copy", "fit", "center-crop", "pad"}:
        raise InvalidPrepareConfig(f"Unsupported prepare mode: {config.mode}")
    if config.image_format not in {"original", "png", "jpg", "webp"}:
        raise InvalidPrepareConfig(f"Unsupported image format: {config.image_format}")
    if config.prompt_type not in {"tags", "natural", "caption", "all"}:
        raise InvalidPrepareConfig(f"Unsupported prompt type: {config.prompt_type}")
    if config.mode == "copy" and config.image_format != "original":
        raise InvalidPrepareConfig("copy mode requires image_format='original'")
    if config.mode == "copy" and (config.width is not None or config.height is not None):
        raise InvalidPrepareConfig("copy mode does not accept width or height")
    if config.mode != "copy" and config.width is None and config.height is None:
        raise InvalidPrepareConfig(f"{config.mode} mode requires width or height")
    for label, value in (("width", config.width), ("height", config.height)):
        if value is not None and value <= 0:
            raise InvalidPrepareConfig(f"{label} must be greater than zero")


def _prepared_filename(source: Path, config: PrepareConfig) -> str:
    if config.image_format == "original":
        return source.name
    return f"{source.stem}{FORMAT_EXTENSIONS[config.image_format]}"


def _prepare_image(
    source: Path,
    target: Path,
    config: PrepareConfig,
    watermark_remover: WatermarkRemover | None,
) -> None:
    if watermark_remover is not None:
        _prepare_watermark_cleaned_image(source, target, config, watermark_remover)
        return
    if config.mode == "copy":
        shutil.copy2(source, target)
        return

    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        _reject_jpg_alpha(source, image, config)
        output = _transform_image(image, config)
        save_format = _save_format(source, config)
        if save_format == "JPEG" and output.mode != "RGB":
            output = output.convert("RGB")
        output.save(target, format=save_format)


def _prepare_watermark_cleaned_image(
    source: Path,
    target: Path,
    config: PrepareConfig,
    watermark_remover: WatermarkRemover,
) -> None:
    if config.mode == "copy":
        watermark_remover.process_file(source, target)
        return
    with tempfile.TemporaryDirectory(prefix="lorakit-watermark-") as temporary_directory:
        cleaned = Path(temporary_directory) / source.name
        watermark_remover.process_file(source, cleaned)
        _prepare_image(cleaned, target, config, None)


def _transform_image(image: Image.Image, config: PrepareConfig) -> Image.Image:
    if config.mode == "fit":
        return _fit(image, config)
    width, height = _box_dimensions(config)
    if config.mode == "center-crop":
        return ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    if config.mode == "pad":
        resized = _fit(image, PrepareConfig(mode="fit", width=width, height=height))
        background = _new_background(resized, width, height)
        left = (width - resized.width) // 2
        top = (height - resized.height) // 2
        background.paste(resized, (left, top), resized if resized.mode == "RGBA" else None)
        return background
    raise InvalidPrepareConfig(f"Unsupported prepare mode: {config.mode}")


def _fit(image: Image.Image, config: PrepareConfig) -> Image.Image:
    width = config.width
    height = config.height
    if width is None and height is None:
        raise InvalidPrepareConfig("fit mode requires width or height")
    if width is None:
        scale = height / image.height
        width = max(1, round(image.width * scale))
    if height is None:
        scale = width / image.width
        height = max(1, round(image.height * scale))
    scale = min(width / image.width, height / image.height)
    target_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    return image.resize(target_size, Image.Resampling.LANCZOS)


def _box_dimensions(config: PrepareConfig) -> tuple[int, int]:
    if config.width is None and config.height is None:
        raise InvalidPrepareConfig(f"{config.mode} mode requires width or height")
    if config.width is None:
        return config.height, config.height
    if config.height is None:
        return config.width, config.width
    return config.width, config.height


def _new_background(image: Image.Image, width: int, height: int) -> Image.Image:
    if image.mode == "RGBA":
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))
    return Image.new("RGB", (width, height), (255, 255, 255))


def _save_format(source: Path, config: PrepareConfig) -> str:
    if config.image_format == "png":
        return "PNG"
    if config.image_format == "jpg":
        return "JPEG"
    if config.image_format == "webp":
        return "WEBP"
    suffix = source.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "JPEG"
    if suffix == ".png":
        return "PNG"
    if suffix == ".webp":
        return "WEBP"
    if suffix == ".bmp":
        return "BMP"
    if suffix == ".gif":
        return "GIF"
    raise InvalidPrepareConfig(f"Cannot preserve unsupported image format: {source}")


def _reject_jpg_alpha(source: Path, image: Image.Image, config: PrepareConfig) -> None:
    if config.image_format != "jpg":
        return
    has_alpha = image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    )
    if has_alpha:
        raise InvalidPrepareConfig(f"Cannot convert transparent image to jpg: {source}")
