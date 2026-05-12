"""Shared backend contracts."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BackendSpec:
    prepared_dir: Path
    output_dir: Path
    model: str
    resolution: int
    rank: int
    steps: int
    learning_rate: float
    batch_size: int
    gradient_accumulation: int
    mixed_precision: str


@dataclass(frozen=True)
class BackendResult:
    model_path: Path
