"""Public Python API for lorakit."""

from __future__ import annotations

from pathlib import Path

import lorakit.candidates as candidates_module
import lorakit.clean as clean_module
import lorakit.datasets as datasets_module
import lorakit.importers as importers_module
import lorakit.models as models_module
import lorakit.prepare as prepare_module
import lorakit.training as training_module
from lorakit.errors import ImporterMissing
from lorakit.paths import Paths
from lorakit.types import (
    Candidate,
    CleanResult,
    DatasetStatus,
    DatasetSummary,
    ImportResult,
    ModelInfo,
    ModelResolution,
    PrepareConfig,
    TrainingResult,
    TrainingSpec,
)


class Project:
    """Facade for working with a lorakit data directory."""

    def __init__(self, data_dir: str | Path = "./data"):
        self.paths = Paths(Path(data_dir))
        self.paths.ensure()
        self.candidates = _Candidates(self.paths)
        self.datasets = _Datasets(self.paths)
        self.models = _Models(self.paths)

    def clean(self, *, apply: bool = False) -> CleanResult:
        return clean_module.clean(self.paths, apply=apply)

    def train(self, spec: TrainingSpec) -> TrainingResult:
        return training_module.train(self.paths, spec)


class _Candidates:
    def __init__(self, paths: Paths):
        self._paths = paths

    def list(self) -> list[Candidate]:
        return candidates_module.list_all(self._paths)

    def show(self, stem: str) -> dict[str, object]:
        return candidates_module.show(self._paths, stem)

    def import_from(
        self,
        source: str,
        args: list[str],
        *,
        overwrite: bool = False,
    ) -> ImportResult:
        if source not in importers_module.REGISTRY:
            raise ImporterMissing(f"Unknown importer: {source}")
        return importers_module.REGISTRY[source].run(
            self._paths,
            args,
            overwrite=overwrite,
        )


class _Datasets:
    def __init__(self, paths: Paths):
        self._paths = paths

    def create(self, name: str) -> Path:
        return datasets_module.create(self._paths, name)

    def delete(self, name: str) -> None:
        datasets_module.delete(self._paths, name)

    def rename(self, old: str, new: str) -> None:
        datasets_module.rename(self._paths, old, new)

    def list(self) -> list[DatasetSummary]:
        return datasets_module.list_all(self._paths)

    def status(self, name: str) -> DatasetStatus:
        return datasets_module.status(self._paths, name)

    def stage(self, dataset: str, image: str | Path, *, symlink: bool = False) -> Path:
        return datasets_module.stage(self._paths, dataset, image, symlink=symlink)

    def stage_all(self, dataset: str, *, symlink: bool = False) -> list[Path]:
        return datasets_module.stage_all(self._paths, dataset, symlink=symlink)

    def unstage(self, dataset: str, image: str | Path) -> Path:
        return datasets_module.unstage(self._paths, dataset, image)

    def prepare(self, dataset: str, config: PrepareConfig) -> Path:
        return prepare_module.prepare(self._paths, dataset, config)


class _Models:
    def __init__(self, paths: Paths):
        self._paths = paths

    def list(self) -> list[ModelInfo]:
        return models_module.list_all(self._paths)

    def search(
        self,
        query: str,
        *,
        task: str = "text-to-image",
        limit: int = 20,
    ) -> list[dict[str, object]]:
        return models_module.search(query, task=task, limit=limit)

    def fetch(self, repo_id: str, *, name: str | None = None) -> Path:
        return models_module.fetch(self._paths, repo_id, name=name)

    def remove(self, name: str) -> Path:
        return models_module.remove(self._paths, name)

    def resolve(self, model: str) -> ModelResolution:
        return models_module.resolve(self._paths, model)


__all__ = [
    "PrepareConfig",
    "Project",
    "TrainingSpec",
    "TrainingResult",
]
