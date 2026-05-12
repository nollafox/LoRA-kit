"""Dataset staging and status operations."""

import json
import shutil
from pathlib import Path

import lorakit.candidates as candidates
from lorakit.errors import DatasetExists, DatasetNotFound, LorakitError, MissingMetadata
from lorakit.manifest import MANIFEST_NAME, read_manifest
from lorakit.paths import Paths
from lorakit.types import DatasetStatus, DatasetSummary


def create(paths: Paths, name: str) -> Path:
    paths.ensure()
    _validate_dataset_name(name)
    target = paths.staged_for(name)
    if target.exists():
        raise DatasetExists(f"Dataset already exists: {name}")
    target.mkdir(parents=True)
    return target


def delete(paths: Paths, name: str) -> None:
    dataset_dir = _require_dataset(paths, name)
    shutil.rmtree(dataset_dir)


def rename(paths: Paths, old: str, new: str) -> None:
    dataset_dir = _require_dataset(paths, old)
    _validate_dataset_name(new)
    target = paths.staged_for(new)
    if target.exists():
        raise DatasetExists(f"Dataset already exists: {new}")
    dataset_dir.rename(target)


def list_all(paths: Paths) -> list[DatasetSummary]:
    paths.ensure()
    summaries: list[DatasetSummary] = []
    for dataset_dir in sorted(path for path in paths.staged.iterdir() if path.is_dir()):
        summaries.append(
            DatasetSummary(
                name=dataset_dir.name,
                staged_images=len(_staged_images(dataset_dir)),
            )
        )
    return summaries


def status(paths: Paths, name: str) -> DatasetStatus:
    dataset_dir = _require_dataset(paths, name)
    staged_images = _staged_images(dataset_dir)
    overrides = [path for path in dataset_dir.iterdir() if path.suffix == ".json"]
    missing_metadata = sum(
        1 for image in staged_images if resolve_metadata(paths, name, image.stem) is None
    )
    artifact_dir = paths.artifacts_for(name)
    artifacts = (
        len([path for path in artifact_dir.iterdir() if path.is_dir()])
        if artifact_dir.exists()
        else 0
    )
    return DatasetStatus(
        name=name,
        staged_images=len(staged_images),
        overrides=len(overrides),
        missing_metadata=missing_metadata,
        prepared=_prepared_is_valid(paths, name),
        artifacts=artifacts,
    )


def stage(paths: Paths, dataset: str, image: str | Path, *, symlink: bool = False) -> Path:
    dataset_dir = _require_dataset(paths, dataset)
    source = Path(image)
    if not source.exists():
        candidate_image = candidates.image_for_stem(paths, str(image))
        if candidate_image is None:
            raise LorakitError(f"Image not found: {image}")
        source = candidate_image
    if not candidates.is_image(source):
        raise LorakitError(f"Not a supported image file: {source}")

    target = dataset_dir / source.name
    if target.exists() or target.is_symlink():
        target.unlink()
    if symlink:
        target.symlink_to(source.resolve())
    else:
        shutil.copy2(source, target)
    return target


def unstage(paths: Paths, dataset: str, image: str | Path) -> Path:
    dataset_dir = _require_dataset(paths, dataset)
    target = _resolve_staged_target(dataset_dir, image)
    target.unlink()
    return target


def resolve_metadata(paths: Paths, dataset: str, stem: str) -> Path | None:
    staged_metadata = paths.staged_for(dataset) / f"{stem}.json"
    if staged_metadata.exists():
        return staged_metadata
    candidate_metadata = paths.candidates / f"{stem}.json"
    if candidate_metadata.exists():
        return candidate_metadata
    return None


def require_metadata(paths: Paths, dataset: str, stem: str) -> Path:
    metadata = resolve_metadata(paths, dataset, stem)
    if metadata is None:
        raise MissingMetadata(f"Missing metadata for staged image: {dataset}/{stem}")
    return metadata


def _require_dataset(paths: Paths, name: str) -> Path:
    paths.ensure()
    _validate_dataset_name(name)
    dataset_dir = paths.staged_for(name)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise DatasetNotFound(f"Dataset not found: {name}")
    return dataset_dir


def _validate_dataset_name(name: str) -> None:
    if name in {"", ".", ".."} or "/" in name or "\\" in name:
        raise LorakitError(f"Invalid dataset name: {name}")


def _staged_images(dataset_dir: Path) -> list[Path]:
    return sorted(path for path in dataset_dir.iterdir() if candidates.is_image(path))


def _resolve_staged_target(dataset_dir: Path, image: str | Path) -> Path:
    raw = Path(image)
    direct = dataset_dir / raw.name
    if direct.exists() or direct.is_symlink():
        return direct
    matches = [path for path in _staged_images(dataset_dir) if path.stem == str(image)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise LorakitError(f"Multiple staged images match: {image}")
    raise LorakitError(f"Staged image not found: {image}")


def _prepared_is_valid(paths: Paths, name: str) -> bool:
    prepared_dir = paths.prepared_for(name)
    manifest_path = prepared_dir / MANIFEST_NAME
    config_path = prepared_dir / "config.json"
    if not prepared_dir.exists() or not manifest_path.exists() or not config_path.exists():
        return False
    try:
        rows = read_manifest(manifest_path)
    except (json.JSONDecodeError, MissingMetadata):
        return False
    for row in rows:
        if (
            "image" not in row
            or "caption" not in row
            or "tags" not in row
            or not isinstance(row["image"], str)
            or not isinstance(row["caption"], str)
            or not isinstance(row["tags"], list)
        ):
            return False
        image_path = prepared_dir / str(row["image"])
        if not image_path.exists():
            return False
    return True
