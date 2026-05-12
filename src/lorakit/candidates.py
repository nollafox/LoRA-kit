"""Candidate image and metadata operations."""

from pathlib import Path
from typing import Iterable

from lorakit.errors import CandidateNotFound, LorakitError
from lorakit.manifest import load_metadata
from lorakit.paths import Paths
from lorakit.types import Candidate


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_all(paths: Paths) -> list[Candidate]:
    paths.ensure()
    files = [path for path in paths.candidates.iterdir() if path.is_file()]
    stems = {path.stem for path in files if is_image(path) or path.suffix == ".json"}
    return [_candidate_for_stem(paths.candidates, stem) for stem in sorted(stems)]


def show(paths: Paths, stem: str) -> dict[str, object]:
    paths.ensure()
    candidate = _candidate_for_stem(paths.candidates, stem)
    if candidate.metadata is None:
        raise CandidateNotFound(f"Candidate metadata not found: {stem}")
    return load_metadata(candidate.metadata)


def image_for_stem(paths: Paths, stem: str) -> Path | None:
    paths.ensure()
    images = _images_for_stem(paths.candidates, stem)
    if len(images) > 1:
        joined = ", ".join(str(path) for path in images)
        raise LorakitError(f"Candidate has multiple images for stem '{stem}': {joined}")
    if len(images) == 0:
        return None
    return images[0]


def _candidate_for_stem(directory: Path, stem: str) -> Candidate:
    images = _images_for_stem(directory, stem)
    if len(images) > 1:
        joined = ", ".join(str(path) for path in images)
        raise LorakitError(f"Candidate has multiple images for stem '{stem}': {joined}")
    image = images[0] if images else None
    metadata = directory / f"{stem}.json"
    return Candidate(stem=stem, image=image, metadata=metadata if metadata.exists() else None)


def _images_for_stem(directory: Path, stem: str) -> list[Path]:
    return sorted(
        path
        for path in _iter_files(directory)
        if path.stem == stem and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _iter_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return []
    return (path for path in directory.iterdir() if path.is_file())
