"""Orphan detection and deletion."""

from pathlib import Path

import lorakit.candidates as candidates
import lorakit.datasets as datasets
from lorakit.paths import Paths
from lorakit.types import CleanResult, Orphan


def find_orphans(paths: Paths) -> list[Orphan]:
    paths.ensure()
    orphans: list[Orphan] = []
    orphans.extend(_candidate_orphans(paths))
    orphans.extend(_staged_orphans(paths))
    return sorted(orphans, key=lambda orphan: str(orphan.path))


def clean(paths: Paths, *, apply: bool = False) -> CleanResult:
    orphans = find_orphans(paths)
    deleted: list[Path] = []
    if apply:
        for orphan in orphans:
            orphan.path.unlink()
            deleted.append(orphan.path)
    return CleanResult(orphans=orphans, deleted=deleted)


def _candidate_orphans(paths: Paths) -> list[Orphan]:
    found: list[Orphan] = []
    for candidate in candidates.list_all(paths):
        if candidate.image is not None and candidate.metadata is None:
            found.append(Orphan(candidate.image, "candidate image has no JSON sibling"))
        if candidate.metadata is not None and candidate.image is None:
            found.append(Orphan(candidate.metadata, "candidate JSON has no image sibling"))
    return found


def _staged_orphans(paths: Paths) -> list[Orphan]:
    found: list[Orphan] = []
    for dataset_dir in sorted(path for path in paths.staged.iterdir() if path.is_dir()):
        dataset_name = dataset_dir.name
        staged_images = {path.stem for path in dataset_dir.iterdir() if candidates.is_image(path)}
        for path in sorted(dataset_dir.iterdir()):
            if candidates.is_image(path) and datasets.resolve_metadata(
                paths, dataset_name, path.stem
            ) is None:
                found.append(Orphan(path, "staged image has no metadata"))
            if path.suffix == ".json" and path.stem not in staged_images:
                found.append(Orphan(path, "staged JSON has no matching staged image"))
    return found
