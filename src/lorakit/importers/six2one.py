"""six2one importer for candidate image and metadata pairs."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import lorakit.candidates as candidates
from lorakit.errors import ImporterMissing, LorakitError
from lorakit.manifest import load_metadata, normalize_tags, write_json
from lorakit.paths import Paths
from lorakit.types import ImportResult


DOCS_URL = "https://github.com/nollafox/six2one"


def run(paths: Paths, args: list[str], *, overwrite: bool = False) -> ImportResult:
    paths.ensure()
    if shutil.which("621") is None:
        raise ImporterMissing(
            "six2one is not installed. Install it with `python -m pip install six2one`; "
            f"docs: {DOCS_URL}"
        )

    with tempfile.TemporaryDirectory(prefix="lorakit-six2one-") as tmp:
        tmp_path = Path(tmp)
        command = ["621", *args, "--out", str(tmp_path)]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            raise LorakitError(
                f"six2one import failed with exit code {error.returncode}"
            ) from error
        return import_pairs(
            paths,
            tmp_path,
            overwrite=overwrite,
            source_site=_source_site(args),
        )


def import_pairs(
    paths: Paths,
    source_dir: Path,
    *,
    overwrite: bool = False,
    source_site: str = "e621",
) -> ImportResult:
    paths.ensure()
    imported: list[Path] = []
    skipped: list[Path] = []
    files = [path for path in source_dir.rglob("*") if path.is_file()]
    stems = sorted(
        {path.stem for path in files if candidates.is_image(path) or path.suffix == ".json"}
    )

    for stem in stems:
        image = _single_image(source_dir, stem)
        metadata = _single_metadata(source_dir, stem)
        if image is None or metadata is None:
            continue
        image_target = paths.candidates / image.name
        metadata_target = paths.candidates / metadata.name

        if image_target.exists() and not overwrite:
            skipped.append(image_target)
        else:
            shutil.copy2(image, image_target)
            imported.append(image_target)

        if metadata_target.exists() and not overwrite:
            skipped.append(metadata_target)
        else:
            write_json(
                metadata_target,
                _candidate_metadata(metadata, source_site=source_site),
            )
            imported.append(metadata_target)
    return ImportResult(imported=imported, skipped=skipped)


def _single_image(source_dir: Path, stem: str) -> Path | None:
    matches = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.stem == stem and candidates.is_image(path)
    )
    if len(matches) > 1:
        raise LorakitError(f"Importer produced multiple images for stem '{stem}'")
    if len(matches) == 0:
        return None
    return matches[0]


def _single_metadata(source_dir: Path, stem: str) -> Path | None:
    matches = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.stem == stem and path.suffix == ".json"
    )
    if len(matches) > 1:
        raise LorakitError(f"Importer produced multiple JSON files for stem '{stem}'")
    if len(matches) == 0:
        return None
    return matches[0]


def _candidate_metadata(source_metadata: Path, *, source_site: str) -> dict[str, object]:
    raw_metadata = load_metadata(source_metadata)
    if "id" not in raw_metadata or not isinstance(raw_metadata["id"], int):
        raise LorakitError(f"six2one metadata is missing integer id: {source_metadata}")
    if "tags" not in raw_metadata:
        raise LorakitError(f"six2one metadata is missing tags: {source_metadata}")
    return {
        "tags": normalize_tags(raw_metadata["tags"], source_metadata),
        "metadata": {
            "source": source_site,
            "post_id": raw_metadata["id"],
        },
    }


def _source_site(args: list[str]) -> str:
    for index, arg in enumerate(args):
        if arg == "--site" and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith("--site="):
            return arg.split("=", maxsplit=1)[1]
    return "e621"
