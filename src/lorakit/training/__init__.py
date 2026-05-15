"""Training orchestration and artifact archiving."""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import lorakit.models as models
import lorakit.prepare as prepare
from lorakit.errors import LorakitError, MissingMetadata
from lorakit.manifest import MANIFEST_NAME, read_manifest, write_json
from lorakit.paths import Paths
from lorakit.training.backends import get_backend
from lorakit.training.backends.types import BackendSpec
from lorakit.types import ModelResolution, TrainingResult, TrainingSpec


CONFIG_FILE_NAME = "config.json"
MODEL_ARTIFACT_NAME = "model.safetensors"
PROBE_LOG_NAME = "context-probes.jsonl"
PROBE_SUMMARY_NAME = "context-probes-summary.json"
RUN_NAME_PREFIX = "run-"
RUN_NAME_INDEX_WIDTH = 3
WORKING_DIR_NAME = "working"


def train(paths: Paths, spec: TrainingSpec) -> TrainingResult:
    paths.ensure()
    resolved_model = models.resolve(paths, spec.model)
    prepared_dir = paths.prepared_for(spec.dataset)
    run_name = spec.run_name if spec.run_name is not None else _next_run_name(paths, spec.dataset)
    artifact_dir = paths.artifacts_for(spec.dataset) / run_name
    plan = _plan(paths, spec, run_name, artifact_dir, resolved_model)

    if spec.dry_run:
        return TrainingResult(
            artifact_dir=artifact_dir,
            run_name=run_name,
            dry_run=True,
            plan=plan,
        )

    if not spec.no_prepare:
        prepared_dir = prepare.prepare(paths, spec.dataset, spec.prepare)
    else:
        _assert_prepared_dataset_usable(prepared_dir)

    if artifact_dir.exists():
        raise LorakitError(f"Artifact directory already exists: {artifact_dir}")
    working_dir = artifact_dir / WORKING_DIR_NAME
    artifact_dir.mkdir(parents=True)
    write_json(artifact_dir / CONFIG_FILE_NAME, plan)

    backend_spec = _backend_spec(
        prepared_dir=prepared_dir,
        output_dir=working_dir,
        model=_model_reference(resolved_model),
        spec=spec,
    )
    completed = False
    try:
        backend_result = get_backend(spec.backend).train(backend_spec)
        archived_artifacts = _archive_backend_result(
            prepared_dir,
            artifact_dir,
            backend_result.model_path,
            backend_result.artifact_paths,
        )
        completed = True
    finally:
        if not completed and artifact_dir.exists():
            shutil.rmtree(artifact_dir)

    return TrainingResult(
        artifact_dir=artifact_dir,
        run_name=run_name,
        probe_log=archived_artifacts.get(PROBE_LOG_NAME),
        probe_summary=archived_artifacts.get(PROBE_SUMMARY_NAME),
    )


def _backend_spec(
    *,
    prepared_dir: Path,
    output_dir: Path,
    model: str,
    spec: TrainingSpec,
) -> BackendSpec:
    return BackendSpec(
        prepared_dir=prepared_dir,
        output_dir=output_dir,
        model=model,
        resolution=spec.resolution,
        rank=spec.rank,
        steps=spec.steps,
        learning_rate=spec.learning_rate,
        batch_size=spec.batch_size,
        gradient_accumulation=spec.gradient_accumulation,
        mixed_precision=spec.mixed_precision,
    )


def _next_run_name(paths: Paths, dataset: str) -> str:
    artifact_root = paths.artifacts_for(dataset)
    if not artifact_root.exists():
        return f"{RUN_NAME_PREFIX}{1:0{RUN_NAME_INDEX_WIDTH}d}"

    highest = 0
    for path in artifact_root.iterdir():
        if not path.is_dir() or not path.name.startswith(RUN_NAME_PREFIX):
            continue
        suffix = path.name.removeprefix(RUN_NAME_PREFIX)
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return f"{RUN_NAME_PREFIX}{highest + 1:0{RUN_NAME_INDEX_WIDTH}d}"


def _plan(
    paths: Paths,
    spec: TrainingSpec,
    run_name: str,
    artifact_dir: Path,
    resolved_model: ModelResolution,
) -> dict[str, object]:
    train_config = asdict(spec)
    train_config.pop("prepare")
    return {
        "dataset": spec.dataset,
        "prepare_would_run": not spec.no_prepare,
        "prepare_config": asdict(spec.prepare),
        "train_config": train_config,
        "resolved_model": _jsonable(asdict(resolved_model)),
        "run_name": run_name,
        "artifact_dir": str(artifact_dir),
        "prepared_dir": str(paths.prepared_for(spec.dataset)),
    }


def _assert_prepared_dataset_usable(prepared_dir: Path) -> None:
    config_path = prepared_dir / CONFIG_FILE_NAME
    manifest_path = prepared_dir / MANIFEST_NAME
    if not prepared_dir.exists():
        raise LorakitError(f"Prepared dataset directory does not exist: {prepared_dir}")
    if not config_path.exists():
        raise LorakitError(f"Prepared dataset is missing config file: {config_path}")
    if not manifest_path.exists():
        raise LorakitError(f"Prepared dataset is missing manifest file: {manifest_path}")

    rows = _read_prepared_manifest(manifest_path)
    if not rows:
        raise LorakitError(f"Prepared dataset manifest has no rows: {manifest_path}")
    for index, row in enumerate(rows):
        _assert_prepared_manifest_row(prepared_dir=prepared_dir, row=row, index=index)


def _read_prepared_manifest(manifest_path: Path) -> list[dict[str, object]]:
    try:
        rows = read_manifest(manifest_path)
    except (json.JSONDecodeError, MissingMetadata) as exc:
        raise LorakitError(f"Prepared dataset manifest is invalid: {manifest_path}") from exc
    if not isinstance(rows, list):
        raise LorakitError(f"Prepared dataset manifest must be a list: {manifest_path}")
    return rows


def _assert_prepared_manifest_row(
    *,
    prepared_dir: Path,
    row: object,
    index: int,
) -> None:
    if not isinstance(row, dict):
        raise LorakitError(f"Prepared manifest row {index} must be an object")
    image = row.get("image")
    caption = row.get("caption")
    tags = row.get("tags")
    if not isinstance(image, str) or image == "":
        raise LorakitError(f"Prepared manifest row {index} is missing image path")
    if not isinstance(caption, str):
        raise LorakitError(f"Prepared manifest row {index} is missing caption")
    if not isinstance(tags, list):
        raise LorakitError(f"Prepared manifest row {index} is missing tags")
    image_path = prepared_dir / image
    if not image_path.exists():
        raise LorakitError(f"Prepared manifest row {index} image does not exist: {image_path}")


def _model_reference(resolved_model: ModelResolution) -> str:
    if resolved_model.path is not None:
        return str(resolved_model.path)
    if resolved_model.repo_id is None:
        raise LorakitError(f"Model could not be resolved: {resolved_model.requested}")
    return resolved_model.repo_id


def _archive_backend_result(
    prepared_dir: Path,
    artifact_dir: Path,
    backend_model_path: Path,
    backend_artifact_paths: tuple[Path, ...],
) -> dict[str, Path]:
    if not backend_model_path.exists():
        raise LorakitError(f"Training backend did not produce model file: {backend_model_path}")
    shutil.copytree(prepared_dir, artifact_dir / "dataset")
    shutil.copy2(backend_model_path, artifact_dir / MODEL_ARTIFACT_NAME)
    archived_artifacts: dict[str, Path] = {}
    for backend_artifact_path in backend_artifact_paths:
        if not backend_artifact_path.exists():
            raise LorakitError(f"Training backend artifact is missing: {backend_artifact_path}")
        destination = artifact_dir / backend_artifact_path.name
        shutil.copy2(backend_artifact_path, destination)
        archived_artifacts[backend_artifact_path.name] = destination
    shutil.rmtree(artifact_dir / WORKING_DIR_NAME)
    return archived_artifacts


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
