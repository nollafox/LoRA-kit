"""Training backend registry."""

from importlib import import_module
from types import MappingProxyType
from typing import Protocol

from lorakit.errors import LorakitError
from lorakit.training.backends.types import BackendResult, BackendSpec


class Backend(Protocol):
    def train(self, spec: BackendSpec) -> BackendResult:
        """Train a LoRA model and return the produced weights."""


BACKEND_MODULES = MappingProxyType(
    {
        "diffusers": "lorakit.training.backends.diffusers",
    }
)


def get_backend(name: str) -> Backend:
    if name not in BACKEND_MODULES:
        raise LorakitError(f"Unknown training backend: {name}")
    try:
        return import_module(BACKEND_MODULES[name])
    except ModuleNotFoundError as exc:
        raise LorakitError(
            f"Training backend '{name}' is missing dependency: {exc.name}"
        ) from exc
