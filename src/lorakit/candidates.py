"""Candidate image and metadata operations."""

from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

import lorakit.models as models_module
from lorakit.errors import CandidateNotFound, LorakitError
from lorakit.manifest import load_metadata, normalize_tags
from lorakit.paths import Paths
from lorakit.tools.tagging import (
    ImageTagger,
    TagRequest,
    TaggingResult,
    build_tagger,
    merge_caption,
    merge_tags,
    metadata_has_tags,
)
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
    presets: list[str] | tuple[str, ...] | None = None,
    limit: int | None = None,
    tagger: ImageTagger | None = None,
    quiet: bool = False,
) -> list[TagResult]:
    paths.ensure()
    active_tagger = tagger
    results: list[TagResult] = []
    work: list[Candidate] = []
    for candidate in list_all(paths):
        if candidate.image is None:
            continue
        if limit is not None and len(work) >= limit:
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
        work.append(candidate)

    if len(work) == 0:
        return results

    old_model_dir = models_module.MODEL_DIR
    old_cache_dir = models_module.HF_CACHE_DIR
    models_module.MODEL_DIR = paths.models
    models_module.HF_CACHE_DIR = paths.huggingface_cache
    try:
        return _tag_candidates(
            paths,
            work,
            results,
            active_tagger or build_tagger(presets=presets),
            quiet=quiet,
        )
    finally:
        models_module.MODEL_DIR = old_model_dir
        models_module.HF_CACHE_DIR = old_cache_dir


def _tag_candidates(
    paths: Paths,
    work: list[Candidate],
    results: list[TagResult],
    active_tagger: ImageTagger,
    *,
    quiet: bool,
) -> list[TagResult]:
    tag_requests = [
        TagRequest(candidate.image, _existing_tags(paths.candidates / f"{candidate.stem}.json"))
        for candidate in work
        if candidate.image is not None
    ]
    if hasattr(active_tagger, "results_for_many_iteratively"):
        saved_results: list[TagResult | None] = [None for _ in work]

        def save_result(index: int, result: TaggingResult) -> None:
            candidate = work[index]
            metadata_path = paths.candidates / f"{candidate.stem}.json"
            _, added = merge_tags(metadata_path, result.tags)
            merge_caption(metadata_path, result)
            saved_results[index] = TagResult(
                stem=candidate.stem,
                metadata=metadata_path,
                added_tags=added,
                skipped=False,
            )

        active_tagger.results_for_many_iteratively(
            tag_requests,
            quiet=quiet,
            on_result=save_result,
        )
        results.extend(result for result in saved_results if result is not None)
        return results
    if hasattr(active_tagger, "results_for_many_with_progress"):
        tagging_results = active_tagger.results_for_many_with_progress(
            tag_requests,
            quiet=quiet,
        )
    elif hasattr(active_tagger, "results_for_many"):
        tagging_results = active_tagger.results_for_many(tag_requests)
    elif hasattr(active_tagger, "tags_for_many"):
        tagging_results = [
            TaggingResult(tags=tags) for tags in active_tagger.tags_for_many(tag_requests)
        ]
    else:
        tagging_results = [
            TaggingResult(
                tags=active_tagger.tags_for(
                    request.image_path,
                    context_tags=request.context_tags,
                )
            )
            for request in tag_requests
        ]

    iterator = tqdm(
        zip(work, tagging_results, strict=True),
        total=len(work),
        desc="Captioning candidates",
        unit="image",
        dynamic_ncols=True,
        leave=False,
        disable=quiet or len(work) == 0,
    )
    for candidate, result in iterator:
        metadata_path = paths.candidates / f"{candidate.stem}.json"
        iterator.set_postfix_str(candidate.stem, refresh=False)
        _, added = merge_tags(metadata_path, result.tags)
        merge_caption(metadata_path, result)
        results.append(
            TagResult(
                stem=candidate.stem,
                metadata=metadata_path,
                added_tags=added,
                skipped=False,
            )
        )
    return results


def _existing_tags(metadata_path: Path) -> list[str]:
    if not metadata_path.exists():
        return []
    metadata = load_metadata(metadata_path)
    return normalize_tags(metadata.get("tags", []), metadata_path)


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
