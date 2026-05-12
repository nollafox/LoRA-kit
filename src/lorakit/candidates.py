"""Candidate image and metadata operations."""

from pathlib import Path
from typing import Iterable

from lorakit.errors import CandidateNotFound, LorakitError
from lorakit.manifest import load_metadata
from lorakit.paths import Paths
from lorakit.tagging import ImageTagger, build_tagger, merge_tags, metadata_has_tags
from lorakit.types import Candidate, TagResult


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


def tag(
    paths: Paths,
    *,
    all_images: bool = False,
    natural: bool = False,
    limit: int | None = None,
    tagger: ImageTagger | None = None,
) -> list[TagResult]:
    paths.ensure()
    active_tagger = tagger
    results: list[TagResult] = []
    tagged_count = 0
    for candidate in list_all(paths):
        if candidate.image is None:
            continue
        if limit is not None and tagged_count >= limit:
            break
        metadata_path = paths.candidates / f"{candidate.stem}.json"
        if not all_images and metadata_has_tags(metadata_path):
            results.append(
                TagResult(
                    stem=candidate.stem,
                    metadata=metadata_path,
                    added_tags=[],
                    skipped=True,
                )
            )
            continue
        if active_tagger is None:
            active_tagger = build_tagger(natural=natural)
        tags = active_tagger.tags_for(candidate.image)
        _, added = merge_tags(metadata_path, tags)
        results.append(
            TagResult(
                stem=candidate.stem,
                metadata=metadata_path,
                added_tags=added,
                skipped=False,
            )
        )
        tagged_count += 1
    return results


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
