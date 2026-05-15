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
    elif not _prepared_is_usable(prepared_dir):
        raise LorakitError(f"Prepared dataset is missing or invalid: {prepared_dir}")

    if artifact_dir.exists():
        raise LorakitError(f"Artifact directory already exists: {artifact_dir}")
    working_dir = artifact_dir / "working"
    artifact_dir.mkdir(parents=True)
    write_json(artifact_dir / "config.json", plan)

    backend_spec = BackendSpec(
        prepared_dir=prepared_dir,
        output_dir=working_dir,
        model=_model_reference(resolved_model),
        resolution=spec.resolution,
        rank=spec.rank,
        steps=spec.steps,
        learning_rate=spec.learning_rate,
        batch_size=spec.batch_size,
        gradient_accumulation=spec.gradient_accumulation,
        mixed_precision=spec.mixed_precision,
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
        probe_log=archived_artifacts.get("context-probes.jsonl"),
    )


def _next_run_name(paths: Paths, dataset: str) -> str:
    artifact_root = paths.artifacts_for(dataset)
    if not artifact_root.exists():
        return "run-001"
    highest = 0
    for path in artifact_root.iterdir():
        if path.is_dir() and path.name.startswith("run-"):
            suffix = path.name.removeprefix("run-")
            if suffix.isdigit():
                highest = max(highest, int(suffix))
    return f"run-{highest + 1:03d}"


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


def _prepared_is_usable(prepared_dir: Path) -> bool:
    manifest_path = prepared_dir / MANIFEST_NAME
    if (
        not prepared_dir.exists()
        or not (prepared_dir / "config.json").exists()
        or not manifest_path.exists()
    ):
        return False
    try:
        rows = read_manifest(manifest_path)
    except (json.JSONDecodeError, MissingMetadata):
        return False
    return all(
        "image" in row
        and "caption" in row
        and "tags" in row
        and isinstance(row["image"], str)
        and isinstance(row["caption"], str)
        and isinstance(row["tags"], list)
        and (prepared_dir / row["image"]).exists()
        for row in rows
    )


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
    shutil.copy2(backend_model_path, artifact_dir / "model.safetensors")
    archived_artifacts: dict[str, Path] = {}
    for backend_artifact_path in backend_artifact_paths:
        if not backend_artifact_path.exists():
            raise LorakitError(f"Training backend artifact is missing: {backend_artifact_path}")
        destination = artifact_dir / backend_artifact_path.name
        shutil.copy2(backend_artifact_path, destination)
        archived_artifacts[backend_artifact_path.name] = destination
    shutil.rmtree(artifact_dir / "working")
    return archived_artifacts


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
