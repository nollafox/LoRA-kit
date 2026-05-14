"""Project configuration discovery and persistence."""

from dataclasses import dataclass
from pathlib import Path

from lorakit.errors import LorakitError


CONFIG_NAME = "lorakit.yaml"
DEFAULT_DATA_DIR = "data"
DEFAULT_MODELS_DIR = "~/.lorakit/models"
DEFAULT_HF_CACHE_DIR = "~/.lorakit/models/cache"


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    root: Path
    data_dir: Path
    models_dir: Path
    huggingface_cache_dir: Path

    @property
    def candidates_dir(self) -> Path:
        return self.data_dir / "candidates"

    @property
    def staged_dir(self) -> Path:
        return self.data_dir / "staged"

    @property
    def prepared_dir(self) -> Path:
        return self.data_dir / "prepared"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"


def init_project(folder: Path) -> ProjectConfig:
    """Create a lorakit project folder and write its config.

    Raises:
        LorakitError: if the target already has a project config.
    """
    project_root = folder.expanduser().resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    config_path = project_root / CONFIG_NAME
    if config_path.exists():
        raise LorakitError(f"Project already initialized: {config_path}")
    config = ProjectConfig(
        name=project_root.name,
        root=project_root,
        data_dir=project_root / DEFAULT_DATA_DIR,
        models_dir=Path(DEFAULT_MODELS_DIR).expanduser(),
        huggingface_cache_dir=Path(DEFAULT_HF_CACHE_DIR).expanduser(),
    )
    write_config(config_path, config)
    return config


def discover_config(start: Path | None = None) -> ProjectConfig | None:
    """Find the nearest lorakit.yaml walking upward from start."""
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        config_path = directory / CONFIG_NAME
        if config_path.exists():
            return read_config(config_path)
    return None


def default_config(root: Path | None = None) -> ProjectConfig:
    project_root = (root or Path.cwd()).expanduser().resolve()
    return ProjectConfig(
        name=project_root.name,
        root=project_root,
        data_dir=project_root / DEFAULT_DATA_DIR,
        models_dir=project_root / "models",
        huggingface_cache_dir=project_root / "models" / "cache",
    )


def read_config(path: Path) -> ProjectConfig:
    data = _parse_simple_yaml(path)
    paths = _require_mapping(data, "paths", path)
    root = path.parent.resolve()
    name = _string_value(data, "name", root.name, path)
    data_dir = _resolve_config_path(root, _string_value(paths, "data", DEFAULT_DATA_DIR, path))
    models_dir = _resolve_config_path(
        root,
        _string_value(paths, "models", DEFAULT_MODELS_DIR, path),
    )
    hf_cache_dir = _resolve_config_path(
        root,
        _string_value(paths, "huggingface_cache", DEFAULT_HF_CACHE_DIR, path),
    )
    return ProjectConfig(
        name=name,
        root=root,
        data_dir=data_dir,
        models_dir=models_dir,
        huggingface_cache_dir=hf_cache_dir,
    )


def write_config(path: Path, config: ProjectConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "version: 1",
            f"name: {config.name}",
            "paths:",
            f"  data: {_format_path(config.data_dir, config.root)}",
            f"  models: {_format_path(config.models_dir, config.root)}",
            f"  huggingface_cache: {_format_path(config.huggingface_cache_dir, config.root)}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _format_path(path: Path, root: Path) -> str:
    expanded = path.expanduser()
    try:
        return expanded.resolve().relative_to(root).as_posix()
    except ValueError:
        home = Path.home().resolve()
        try:
            return f"~/{expanded.resolve().relative_to(home).as_posix()}"
        except ValueError:
            return expanded.as_posix()


def _resolve_config_path(root: Path, value: str) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw
    return root / raw


def _parse_simple_yaml(path: Path) -> dict[str, object]:
    root: dict[str, object] = {}
    current_mapping: dict[str, object] | None = None
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if raw_line.strip() == "" or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if ":" not in stripped:
            raise LorakitError(f"Invalid config line {line_number}: {path}")
        key, raw_value = stripped.split(":", 1)
        value = raw_value.strip()
        if indent == 0:
            if value == "":
                mapping: dict[str, object] = {}
                root[key] = mapping
                current_mapping = mapping
            else:
                root[key] = _scalar(value)
                current_mapping = None
        elif indent == 2 and current_mapping is not None:
            current_mapping[key] = _scalar(value)
        else:
            raise LorakitError(f"Unsupported config indentation on line {line_number}: {path}")
    return root


def _scalar(value: str) -> object:
    if value.isdigit():
        return int(value)
    return value.strip('"').strip("'")


def _require_mapping(data: dict[str, object], key: str, path: Path) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise LorakitError(f"Config is missing mapping '{key}': {path}")
    return value


def _string_value(
    data: dict[str, object],
    key: str,
    default: str,
    path: Path,
) -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise LorakitError(f"Config value '{key}' must be a string: {path}")
    if value == "":
        raise LorakitError(f"Config value '{key}' cannot be empty: {path}")
    return value
