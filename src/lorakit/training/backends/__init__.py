"""Training backend registry."""

from importlib import import_module
from typing import Protocol

from lorakit.errors import LorakitError
from lorakit.training.backends.types import BackendResult, BackendSpec


class Backend(Protocol):
    def train(self, spec: BackendSpec) -> BackendResult:
        """Train a LoRA model and return the produced weights."""


BACKENDS = {"diffusers": "lorakit.training.backends.diffusers"}


def get_backend(name: str) -> Backend:
    if name not in BACKENDS:
        raise LorakitError(f"Unknown training backend: {name}")
    return import_module(BACKENDS[name])
