"""Importer registry."""

from typing import Protocol

from lorakit.paths import Paths
from lorakit.types import ImportResult
from lorakit.importers import six2one


class Importer(Protocol):
    def run(self, paths: Paths, args: list[str], *, overwrite: bool = False) -> ImportResult:
        """Import files into candidates."""


REGISTRY = {
    "621": six2one,
    "six2one": six2one,
}
