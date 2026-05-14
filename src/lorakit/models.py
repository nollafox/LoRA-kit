"""Model discovery, search, fetch, removal, and resolution."""

import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from lorakit.errors import ModelAmbiguous, ModelNotFound
from lorakit.paths import Paths
from lorakit.tools.watermark import ensure_watermark_models
from lorakit.types import ModelInfo, ModelResolution


MODEL_HOME = Path.home() / ".lorakit"
MODEL_DIR = MODEL_HOME / "models"
HF_CACHE_DIR = MODEL_DIR / "cache"

SD15_REPO_ID = "runwayml/stable-diffusion-v1-5"
SD15_FILENAME = "v1-5-pruned-emaonly.safetensors"
SD15_TARGET_NAME = "stable-diffusion-v1-5.safetensors"

SMILINGWOLF_REPO_ID = "SmilingWolf/wd-vit-tagger-v3"
SMILINGWOLF_MODEL_FILE = "model.onnx"
SMILINGWOLF_TAGS_FILE = "selected_tags.csv"
DEFAULT_PIPELINE_SMILINGWOLF_MODEL = "SmilingWolf/wd-eva02-large-tagger-v3"
FLORENCE_PROMPTGEN_REPO_ID = "Disty0/Florence-2-large-PromptGen-v2.0"
DEFAULT_CAPTION_EDITOR_MODEL = "dphn/Dolphin3.0-Llama3.1-8B"
RAM_PLUS_REPO_ID = "xinyu1205/recognize-anything-plus-model"
RAM_PLUS_FILENAME = "ram_plus_swin_large_14m.pth"
DEFAULT_QWEN_VL_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

CHECKPOINT_EXTENSIONS = {".safetensors", ".ckpt", ".pt"}


def install(paths: Paths, *, with_models: bool = False) -> Path:
    paths.ensure_models()
    if with_models:
        _install_known_models(paths)
    return paths.models


def _install_known_models(paths: Paths) -> None:
    ensure_watermark_models(
        models_dir=paths.models,
        cache_dir=paths.huggingface_cache,
    )
    _install_sd15(paths)
    _snapshot_repo(paths, SMILINGWOLF_REPO_ID)
    _snapshot_repo(paths, DEFAULT_PIPELINE_SMILINGWOLF_MODEL)
    _snapshot_repo(paths, FLORENCE_PROMPTGEN_REPO_ID)
    _snapshot_repo(paths, DEFAULT_CAPTION_EDITOR_MODEL)
    _snapshot_repo(paths, RAM_PLUS_REPO_ID)
    _snapshot_repo(paths, DEFAULT_QWEN_VL_MODEL)


def _install_sd15(paths: Paths) -> Path:
    target = paths.models / SD15_TARGET_NAME
    if target.exists():
        return target
    source = hf_hub_download(
        repo_id=SD15_REPO_ID,
        filename=SD15_FILENAME,
        cache_dir=paths.huggingface_cache,
    )
    shutil.copy2(source, target)
    return target


def _snapshot_repo(paths: Paths, repo_id: str) -> Path:
    target = paths.models / repo_id.rsplit("/", 1)[-1]
    if target.exists():
        return target
    return Path(
        snapshot_download(
            repo_id=repo_id,
            local_dir=target,
            cache_dir=paths.huggingface_cache,
        )
    )


def list_all(paths: Paths) -> list[ModelInfo]:
    paths.ensure_models()
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
    paths.ensure_models()
    filename = _select_safetensors_file(repo_id)
    target_name = _target_filename(filename, name)
    target = paths.models / target_name
    source = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=paths.huggingface_cache,
    )
    shutil.copy2(source, target)
    return target


def _select_safetensors_file(repo_id: str) -> str:
    api = HfApi()
    files = sorted(
        path for path in api.list_repo_files(repo_id=repo_id) if path.endswith(".safetensors")
    )
    if len(files) == 0:
        raise ModelNotFound(f"No .safetensors files found in Hugging Face repo: {repo_id}")
    if len(files) == 1:
        return files[0]
    if not sys.stdin.isatty():
        joined = ", ".join(files)
        raise ModelAmbiguous(
            f"Multiple .safetensors files found in {repo_id}; run interactively to choose: "
            f"{joined}"
        )
    print(f"Multiple .safetensors files found in {repo_id}:")
    for index, filename in enumerate(files, start=1):
        print(f"  {index}. {filename}")
    answer = input(f"Select file [1-{len(files)}]: ")
    if not answer.isdigit():
        raise ModelNotFound(f"Invalid selection: {answer}")
    selected = int(answer)
    if selected < 1 or selected > len(files):
        raise ModelNotFound(f"Selection out of range: {answer}")
    return files[selected - 1]


def _target_filename(source_filename: str, name: str | None) -> str:
    source_path = Path(source_filename)
    if name is None:
        return source_path.name
    target = Path(name)
    if target.suffix == "":
        return f"{name}{source_path.suffix}"
    return target.name


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
    paths.ensure_models()
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
