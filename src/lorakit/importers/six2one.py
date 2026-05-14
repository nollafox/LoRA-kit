"""six2one importer for candidate image and metadata pairs."""

import shutil
import subprocess
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

    cache_dir = paths.project_root / ".lorakit" / "cache" / "621"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = _find_output_path(args) or cache_dir
    command = ["621", *args]
    if "--merge" not in args and not any(arg.startswith("--merge=") for arg in args):
        command.append("--merge")
    if "--out" not in args and not any(arg.startswith("--out=") for arg in args):
        command.extend(["--out", str(output_path)])

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise LorakitError(
            f"six2one import failed with exit code {error.returncode}"
        ) from error

    if _is_six2one_output(output_path):
        return _import_six2one_pairs(
            paths,
            output_path,
            overwrite=overwrite,
            source_site=_source_site(args),
            args=args,
        )

    return import_pairs(
        paths,
        output_path,
        overwrite=overwrite,
        source_site=_source_site(args),
    )


def import_pairs(
    paths: Paths,
    source_dir: Path,
    *,
    overwrite: bool = False,
    source_site: str = "e621",
    only_stems: set[str] | None = None,
) -> ImportResult:
    paths.ensure()
    imported: list[Path] = []
    skipped: list[Path] = []
    files = [path for path in source_dir.rglob("*") if path.is_file()]
    stems = sorted(
        {path.stem for path in files if candidates.is_image(path) or path.suffix == ".json"}
    )
    if only_stems is not None:
        stems = [stem for stem in stems if stem in only_stems]

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
    tags = _filter_e621_tags(raw_metadata["tags"])
    return {
        "tags": normalize_tags(tags, source_metadata),
        "metadata": {
            "source": source_site,
            "post_id": raw_metadata["id"],
        },
    }


def _filter_e621_tags(tags: object) -> object:
    if isinstance(tags, dict):
        return {key: tags[key] for key in ("general", "meta") if key in tags}
    return tags


def _find_output_path(args: list[str]) -> Path | None:
    for index, arg in enumerate(args):
        if arg == "--out" and index + 1 < len(args):
            return Path(args[index + 1])
        if arg.startswith("--out="):
            return Path(arg.split("=", maxsplit=1)[1])
    return None


def _is_six2one_output(source_dir: Path) -> bool:
    return (
        source_dir.exists()
        and source_dir.is_dir()
        and (source_dir / "manifest.json").exists()
        and (source_dir / "images").is_dir()
        and (source_dir / "posts").is_dir()
    )


def _import_six2one_pairs(
    paths: Paths,
    source_dir: Path,
    *,
    overwrite: bool = False,
    source_site: str = "e621",
    args: list[str],
) -> ImportResult:
    paths.ensure()
    imported: list[Path] = []
    skipped: list[Path] = []
    manifest = _load_six2one_manifest(source_dir)
    post_ids = _query_post_ids(source_dir, args, manifest=manifest)

    for post_id in sorted(post_ids):
        image_source = _six2one_image_path(source_dir, post_id, manifest)
        metadata_source = source_dir / "posts" / f"{post_id:012d}.json"
        if image_source is None or not metadata_source.exists():
            continue

        image_target = paths.candidates / image_source.name
        metadata_target = paths.candidates / metadata_source.name

        if image_target.exists() and not overwrite:
            skipped.append(image_target)
        else:
            shutil.copy2(image_source, image_target)
            imported.append(image_target)

        if metadata_target.exists() and not overwrite:
            skipped.append(metadata_target)
        else:
            write_json(
                metadata_target,
                _candidate_metadata(metadata_source, source_site=source_site),
            )
            imported.append(metadata_target)

    return ImportResult(imported=imported, skipped=skipped)


def _query_post_ids(source_dir: Path, args: list[str], manifest: dict[str, object] | None) -> set[int]:
    if manifest is not None:
        post_ids = _post_ids_for_query(source_dir, args, manifest)
        if post_ids:
            return post_ids
    return _complete_six2one_post_ids(source_dir)


def _load_six2one_manifest(source_dir: Path) -> dict[str, object] | None:
    try:
        from six2one.manifest import load_manifest, normalize_manifest
    except ImportError:
        return None

    manifest_path = source_dir / "manifest.json"
    raw_manifest = load_manifest(manifest_path)
    if raw_manifest is None:
        return None
    return normalize_manifest(raw_manifest, source_dir)


def _post_ids_for_query(source_dir: Path, args: list[str], manifest: dict[str, object]) -> set[int]:
    try:
        from six2one.cli import parse_fetch_config
        from six2one.query import compile_query
        from six2one.manifest import query_key_for, query_state
    except ImportError:
        return set()

    config = parse_fetch_config(args)
    query = compile_query(config)
    query_key = query_key_for(query, config.site, config.file_mode)
    queries = manifest.get("queries", {})
    if query_key not in queries:
        return set()

    state = query_state(manifest, query_key)
    seen_ids = state.get("seen_post_ids")
    if not isinstance(seen_ids, list):
        return set()
    return {post_id for post_id in seen_ids if isinstance(post_id, int)}


def _complete_six2one_post_ids(source_dir: Path) -> set[int]:
    images = _ids_from_directory(source_dir / "images")
    posts = _ids_from_directory(source_dir / "posts")
    return images & posts


def _ids_from_directory(directory: Path) -> set[int]:
    if not directory.exists() or not directory.is_dir():
        return set()
    ids: set[int] = set()
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.stem.isdigit():
            ids.add(int(path.stem))
    return ids


def _six2one_image_path(source_dir: Path, post_id: int, manifest: dict[str, object] | None) -> Path | None:
    if manifest is not None:
        post_record = manifest.get("posts", {}).get(str(post_id))
        if isinstance(post_record, dict):
            files = post_record.get("files")
            if isinstance(files, dict):
                for file_record in files.values():
                    if isinstance(file_record, dict):
                        path = file_record.get("path")
                        if isinstance(path, str):
                            candidate = source_dir / path
                            if candidate.exists() and candidate.is_file():
                                return candidate
    matches = sorted((source_dir / "images").glob(f"{post_id:012d}.*"))
    return matches[0] if matches else None


def _source_site(args: list[str]) -> str:
    for index, arg in enumerate(args):
        if arg == "--site" and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith("--site="):
            return arg.split("=", maxsplit=1)[1]
    return "e621"
