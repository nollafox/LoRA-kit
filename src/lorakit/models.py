"""Model discovery, search, fetch, removal, and resolution."""

import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from lorakit.errors import ModelAmbiguous, ModelNotFound
from lorakit.paths import Paths
from lorakit.types import ModelInfo, ModelResolution


CHECKPOINT_EXTENSIONS = {".safetensors", ".ckpt", ".pt"}


def list_all(paths: Paths) -> list[ModelInfo]:
    paths.ensure()
    models: list[ModelInfo] = []
    for path in sorted(paths.models.iterdir()):
        if path.is_file() and path.suffix.lower() in CHECKPOINT_EXTENSIONS:
            models.append(
                ModelInfo(
                    name=path.stem,
                    type="checkpoint",
                    format=path.suffix.lower().lstrip("."),
                    path=path,
                )
            )
        if path.is_dir() and (path / "model_index.json").exists():
            models.append(
                ModelInfo(
                    name=path.name,
                    type="diffusers",
                    format="directory",
                    path=path,
                )
            )
    return models


def search(query: str, *, task: str = "text-to-image", limit: int = 20) -> list[dict[str, object]]:
    api = HfApi()
    results = api.list_models(search=query, pipeline_tag=task, limit=limit)
    return [
        {
            "model_id": model.modelId,
            "author": model.author,
            "downloads": model.downloads,
            "likes": model.likes,
        }
        for model in results
    ]


def fetch(paths: Paths, repo_id: str, *, name: str | None = None) -> Path:
    paths.ensure()
    target_name = name if name is not None else repo_id.rsplit("/", 1)[-1]
    target = paths.models / target_name
    snapshot_download(repo_id=repo_id, local_dir=target)
    return target


def remove(paths: Paths, name: str) -> Path:
    resolved = resolve(paths, name)
    if resolved.source == "huggingface" or resolved.path is None:
        raise ModelNotFound(f"Local model not found: {name}")
    target = resolved.path
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return target


def resolve(paths: Paths, model: str) -> ModelResolution:
    paths.ensure()
    direct = Path(model)
    if direct.exists():
        return ModelResolution(requested=model, source="path", path=direct, repo_id=None)

    model_dir = paths.models / model
    if model_dir.is_dir():
        return ModelResolution(requested=model, source="local", path=model_dir, repo_id=None)

    matches = sorted(
        path
        for path in paths.models.glob(f"{model}.*")
        if path.is_file() and path.suffix.lower() in CHECKPOINT_EXTENSIONS
    )
    if len(matches) > 1:
        joined = ", ".join(str(path) for path in matches)
        raise ModelAmbiguous(f"Model name matches multiple files: {joined}")
    if len(matches) == 1:
        return ModelResolution(requested=model, source="local", path=matches[0], repo_id=None)

    return ModelResolution(requested=model, source="huggingface", path=None, repo_id=model)
