"""Filesystem layout for lorakit projects."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """Resolved locations for a lorakit project rooted beside the data directory."""

    root: Path
    project_directory: Path | None = None
    models_directory: Path | None = None
    huggingface_cache_directory: Path | None = None

    @property
    def project_root(self) -> Path:
        return self.project_directory or self.root.parent

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
        return self.models_directory or default_model_home()

    @property
    def huggingface_cache(self) -> Path:
        return self.huggingface_cache_directory or default_huggingface_cache()

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
            self.artifacts,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def ensure_models(self) -> None:
        """Create configured model storage directories."""
        for path in (self.models, self.huggingface_cache):
            path.mkdir(parents=True, exist_ok=True)


def default_model_home() -> Path:
    return Path.home() / ".lorakit" / "models"


def default_huggingface_cache() -> Path:
    return default_model_home() / "cache"
