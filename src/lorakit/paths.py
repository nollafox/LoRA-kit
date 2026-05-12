"""Filesystem layout for lorakit projects."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """Resolved locations under a lorakit data directory."""

    root: Path

    @property
    def candidates(self) -> Path:
        return self.root / "candidates"

    @property
    def staged(self) -> Path:
        return self.root / "staged"

    @property
    def prepared(self) -> Path:
        return self.root / "prepared"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    def staged_for(self, dataset: str) -> Path:
        return self.staged / dataset

    def prepared_for(self, dataset: str) -> Path:
        return self.prepared / dataset

    def artifacts_for(self, dataset: str) -> Path:
        return self.artifacts / dataset

    def ensure(self) -> None:
        """Create the standard data directory layout."""
        for path in (
            self.candidates,
            self.staged,
            self.prepared,
            self.models,
            self.artifacts,
        ):
            path.mkdir(parents=True, exist_ok=True)
