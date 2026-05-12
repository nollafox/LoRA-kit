"""Shared dataclasses for lorakit domain modules."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Candidate:
    stem: str
    image: Path | None
    metadata: Path | None

    @property
    def is_broken(self) -> bool:
        return self.image is None or self.metadata is None


@dataclass(frozen=True)
class DatasetSummary:
    name: str
    staged_images: int


@dataclass(frozen=True)
class DatasetCreated:
    name: str
    path: Path


@dataclass(frozen=True)
class DatasetStatus:
    name: str
    staged_images: int
    overrides: int
    missing_metadata: int
    prepared: bool
    artifacts: int


@dataclass(frozen=True)
class PrepareConfig:
    mode: Literal["copy", "fit", "center-crop", "pad"] = "fit"
    width: int | None = 512
    height: int | None = 512
    image_format: Literal["png", "jpg", "webp", "original"] = "original"
    trigger: str = ""


@dataclass(frozen=True)
class TrainingSpec:
    dataset: str
    model: str = "sd15"
    backend: str = "diffusers"
    resolution: int = 512
    rank: int = 16
    steps: int = 2000
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation: int = 4
    mixed_precision: str = "fp16"
    run_name: str | None = None
    no_prepare: bool = False
    dry_run: bool = False
    prepare: PrepareConfig = field(default_factory=PrepareConfig)


@dataclass(frozen=True)
class TrainingResult:
    artifact_dir: Path
    run_name: str
    dry_run: bool = False
    plan: dict[str, object] | None = None


@dataclass(frozen=True)
class Orphan:
    path: Path
    reason: str


@dataclass(frozen=True)
class CleanResult:
    orphans: list[Orphan]
    deleted: list[Path]


@dataclass(frozen=True)
class ImportResult:
    imported: list[Path]
    skipped: list[Path]


@dataclass(frozen=True)
class TagResult:
    stem: str
    metadata: Path
    added_tags: list[str]
    skipped: bool = False


@dataclass(frozen=True)
class ModelInfo:
    name: str
    type: Literal["checkpoint", "diffusers"]
    format: str
    path: Path


@dataclass(frozen=True)
class ModelResolution:
    requested: str
    source: Literal["path", "local", "huggingface"]
    path: Path | None
    repo_id: str | None
