"""Command-line interface for lorakit."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from lorakit import PrepareConfig, Project, TrainingSpec
from lorakit.errors import LorakitError
from lorakit.types import (
    Candidate,
    CleanResult,
    DatasetCreated,
    DatasetStatus,
    DatasetSummary,
    ImportResult,
    ModelInfo,
    TagResult,
    TrainingResult,
)


DEFAULT_PREPARE_SIZE = 512
DEFAULT_DATA_DIR = "./data"
DEFAULT_OUTPUT = "text"
DEFAULT_RANK = 16
DEFAULT_STEPS = 2000
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION = 4


@dataclass(frozen=True)
class TrainPreset:
    description: str
    rank: int
    steps: int
    learning_rate: float
    batch_size: int = DEFAULT_BATCH_SIZE
    gradient_accumulation: int = DEFAULT_GRADIENT_ACCUMULATION


TRAIN_PRESETS = {
    "concept": TrainPreset(
        description="simple visual concepts: ears, markings, props, small objects",
        rank=8,
        steps=1200,
        learning_rate=1e-4,
    ),
    "clothing": TrainPreset(
        description="a specific garment or wearable item",
        rank=16,
        steps=2000,
        learning_rate=1e-4,
    ),
    "character": TrainPreset(
        description="balanced character identity training",
        rank=16,
        steps=2200,
        learning_rate=1e-4,
    ),
    "style": TrainPreset(
        description="an artist, style, or aesthetic LoRA",
        rank=32,
        steps=2500,
        learning_rate=5e-5,
    ),
    "clothing-simple": TrainPreset(
        description="hats, collars, simple shirts, glasses, simple jackets",
        rank=8,
        steps=1500,
        learning_rate=1e-4,
    ),
    "clothing-detailed": TrainPreset(
        description="complex outfits, armor, uniforms, accessories, patterns",
        rank=32,
        steps=2800,
        learning_rate=5e-5,
    ),
    "character-lite": TrainPreset(
        description="flexible character LoRAs that should not overfit too hard",
        rank=8,
        steps=1600,
        learning_rate=1e-4,
    ),
    "character-detail": TrainPreset(
        description="detailed identity, markings, outfit, and body features",
        rank=16,
        steps=2600,
        learning_rate=1e-4,
    ),
    "style-soft": TrainPreset(
        description="promptable style influence that should not overpower images",
        rank=16,
        steps=2000,
        learning_rate=5e-5,
    ),
    "style-strong": TrainPreset(
        description="more faithful style capture",
        rank=32,
        steps=3000,
        learning_rate=5e-5,
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    try:
        result = args.handler(args)
    except LorakitError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if result is not None:
        _emit(result, args.output, args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lorakit")
    _add_global_options(parser, root=True)
    subparsers = parser.add_subparsers(dest="command")

    _add_candidates(subparsers)
    _add_clean(subparsers)
    _add_dataset(subparsers)
    _add_models(subparsers)
    _add_train(subparsers)
    return parser


def _add_global_options(parser: argparse.ArgumentParser, *, root: bool = False) -> None:
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR if root else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output",
        choices=["text", "json", "jsonl"],
        default=DEFAULT_OUTPUT if root else argparse.SUPPRESS,
    )


def _add_candidates(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("candidates")
    _add_global_options(parser)
    commands = parser.add_subparsers(dest="candidate_command", required=True)

    import_parser = commands.add_parser("import")
    _add_global_options(import_parser)
    import_parser.add_argument("source")
    import_parser.add_argument("importer_args", nargs=argparse.REMAINDER)
    import_parser.add_argument("--overwrite", action="store_true")
    import_parser.set_defaults(handler=_cmd_candidates_import)

    list_parser = commands.add_parser("list")
    _add_global_options(list_parser)
    list_parser.set_defaults(handler=_cmd_candidates_list)

    show_parser = commands.add_parser("show")
    _add_global_options(show_parser)
    show_parser.add_argument("stem")
    show_parser.set_defaults(handler=_cmd_candidates_show)

    tag_parser = commands.add_parser(
        "tag",
        description="Auto-tag candidate images with SmilingWolf, optionally adding JoyCaption tags.",
    )
    _add_global_options(tag_parser)
    tag_parser.add_argument(
        "--all",
        action="store_true",
        help="tag every candidate image and merge tags into existing metadata",
    )
    tag_parser.add_argument(
        "--natural",
        action="store_true",
        help="also add natural-language scene tags using JoyCaption",
    )
    tag_parser.add_argument(
        "--limit",
        type=int,
        help="maximum number of candidate images to tag in this run",
    )
    tag_parser.set_defaults(handler=_cmd_candidates_tag)


def _add_clean(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("clean")
    _add_global_options(parser)
    parser.add_argument("--apply", action="store_true")
    parser.set_defaults(handler=_cmd_clean)


def _add_dataset(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("dataset")
    _add_global_options(parser)
    commands = parser.add_subparsers(dest="dataset_command", required=True)

    create_parser = commands.add_parser("create")
    _add_global_options(create_parser)
    create_parser.add_argument("name")
    create_parser.set_defaults(handler=_cmd_dataset_create)

    delete_parser = commands.add_parser("delete")
    _add_global_options(delete_parser)
    delete_parser.add_argument("name")
    delete_parser.add_argument("--yes", action="store_true")
    delete_parser.set_defaults(handler=_cmd_dataset_delete)

    rename_parser = commands.add_parser("rename")
    _add_global_options(rename_parser)
    rename_parser.add_argument("old")
    rename_parser.add_argument("new")
    rename_parser.set_defaults(handler=_cmd_dataset_rename)

    stage_parser = commands.add_parser("stage")
    _add_global_options(stage_parser)
    stage_parser.add_argument("dataset")
    stage_parser.add_argument("image", nargs="?")
    stage_parser.add_argument("--symlink", action="store_true")
    stage_parser.add_argument("--all", action="store_true")
    stage_parser.set_defaults(handler=_cmd_dataset_stage)

    unstage_parser = commands.add_parser("unstage")
    _add_global_options(unstage_parser)
    unstage_parser.add_argument("dataset")
    unstage_parser.add_argument("image")
    unstage_parser.set_defaults(handler=_cmd_dataset_unstage)

    list_parser = commands.add_parser("list")
    _add_global_options(list_parser)
    list_parser.set_defaults(handler=_cmd_dataset_list)

    status_parser = commands.add_parser("status")
    _add_global_options(status_parser)
    status_parser.add_argument("name")
    status_parser.set_defaults(handler=_cmd_dataset_status)

    prepare_parser = commands.add_parser("prepare")
    _add_global_options(prepare_parser)
    prepare_parser.add_argument("dataset")
    _add_prepare_options(prepare_parser)
    prepare_parser.set_defaults(handler=_cmd_dataset_prepare)


def _add_models(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("models")
    _add_global_options(parser)
    commands = parser.add_subparsers(dest="models_command", required=True)

    list_parser = commands.add_parser("list")
    _add_global_options(list_parser)
    list_parser.set_defaults(handler=_cmd_models_list)

    search_parser = commands.add_parser("search")
    _add_global_options(search_parser)
    search_parser.add_argument("query")
    search_parser.add_argument("--task", default="text-to-image")
    search_parser.add_argument("--limit", type=int, default=20)
    search_parser.set_defaults(handler=_cmd_models_search)

    fetch_parser = commands.add_parser("fetch")
    _add_global_options(fetch_parser)
    fetch_parser.add_argument("repo_id")
    fetch_parser.add_argument("--name")
    fetch_parser.set_defaults(handler=_cmd_models_fetch)

    remove_parser = commands.add_parser("remove")
    _add_global_options(remove_parser)
    remove_parser.add_argument("name")
    remove_parser.add_argument("--yes", action="store_true")
    remove_parser.set_defaults(handler=_cmd_models_remove)


def _add_train(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "train",
        description="Train a LoRA from a staged dataset.",
        epilog=_train_preset_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_global_options(parser)
    parser.add_argument("dataset")
    parser.add_argument("--model", default="sd15")
    parser.add_argument("--backend", default="diffusers")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--preset",
        choices=sorted(TRAIN_PRESETS),
        help="training preset; individual train flags still override preset values",
    )
    parser.add_argument("--rank", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--mixed-precision", default="fp16")
    parser.add_argument("--run-name")
    parser.add_argument("--no-prepare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    _add_prepare_options(parser, include_size=True)
    parser.set_defaults(handler=_cmd_train)


def _train_preset_help() -> str:
    lines = ["presets:"]
    for name in sorted(TRAIN_PRESETS):
        preset = TRAIN_PRESETS[name]
        lines.append(
            "  "
            f"{name:<18} {preset.description}; "
            f"rank={preset.rank}, steps={preset.steps}, "
            f"lr={preset.learning_rate:g}, batch={preset.batch_size}, "
            f"grad_accum={preset.gradient_accumulation}"
        )
    lines.append("")
    lines.append("Preset values are defaults. Pass --rank, --steps, --learning-rate,")
    lines.append("--batch-size, or --gradient-accumulation to override individual fields.")
    return "\n".join(lines)


def _add_prepare_options(
    parser: argparse.ArgumentParser,
    *,
    include_size: bool = True,
) -> None:
    parser.add_argument("--mode", choices=["copy", "fit", "center-crop", "pad"], default="fit")
    if include_size:
        parser.add_argument("--size", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--trigger", default="")
    parser.add_argument(
        "--image-format",
        choices=["png", "jpg", "webp", "original"],
        default="original",
    )


def _cmd_candidates_import(args: argparse.Namespace) -> Any:
    importer_args = list(args.importer_args)
    importer_args, data_dir, output = _extract_import_global_options(importer_args)
    if data_dir is not None:
        args.data_dir = data_dir
    if output is not None:
        args.output = output
    overwrite = args.overwrite
    if "--overwrite" in importer_args:
        importer_args.remove("--overwrite")
        overwrite = True
    return Project(args.data_dir).candidates.import_from(
        args.source,
        importer_args,
        overwrite=overwrite,
    )


def _extract_import_global_options(
    importer_args: list[str],
) -> tuple[list[str], str | None, str | None]:
    remaining: list[str] = []
    data_dir: str | None = None
    output: str | None = None
    index = 0
    while index < len(importer_args):
        argument = importer_args[index]
        if argument == "--data-dir":
            if index + 1 >= len(importer_args):
                raise LorakitError("--data-dir requires a value")
            data_dir = importer_args[index + 1]
            index += 2
        elif argument.startswith("--data-dir="):
            data_dir = argument.removeprefix("--data-dir=")
            if data_dir == "":
                raise LorakitError("--data-dir requires a value")
            index += 1
        elif argument == "--output":
            if index + 1 >= len(importer_args):
                raise LorakitError("--output requires a value")
            output = _validate_output(importer_args[index + 1])
            index += 2
        elif argument.startswith("--output="):
            output = _validate_output(argument.removeprefix("--output="))
            index += 1
        else:
            remaining.append(argument)
            index += 1
    return remaining, data_dir, output


def _validate_output(value: str) -> str:
    if value not in {"text", "json", "jsonl"}:
        raise LorakitError(f"Invalid --output: {value}")
    return value


def _cmd_candidates_list(args: argparse.Namespace) -> Any:
    return Project(args.data_dir).candidates.list()


def _cmd_candidates_show(args: argparse.Namespace) -> Any:
    return Project(args.data_dir).candidates.show(args.stem)


def _cmd_candidates_tag(args: argparse.Namespace) -> Any:
    return Project(args.data_dir).candidates.tag(
        all_images=args.all,
        natural=args.natural,
        limit=args.limit,
    )


def _cmd_clean(args: argparse.Namespace) -> Any:
    return Project(args.data_dir).clean(apply=args.apply)


def _cmd_dataset_create(args: argparse.Namespace) -> DatasetCreated:
    path = Project(args.data_dir).datasets.create(args.name)
    return DatasetCreated(name=args.name, path=path)


def _cmd_dataset_delete(args: argparse.Namespace) -> None:
    _confirm_or_error(args.yes, f"Delete dataset '{args.name}'?")
    Project(args.data_dir).datasets.delete(args.name)


def _cmd_dataset_rename(args: argparse.Namespace) -> None:
    Project(args.data_dir).datasets.rename(args.old, args.new)


def _cmd_dataset_stage(args: argparse.Namespace) -> Any:
    if args.all and args.image is not None:
        raise LorakitError("dataset stage accepts either an image or --all, not both")
    if args.all:
        return Project(args.data_dir).datasets.stage_all(
            args.dataset,
            symlink=args.symlink,
        )
    if args.image is None:
        raise LorakitError("dataset stage requires an image or --all")
    return Project(args.data_dir).datasets.stage(
        args.dataset,
        args.image,
        symlink=args.symlink,
    )


def _cmd_dataset_unstage(args: argparse.Namespace) -> Any:
    return Project(args.data_dir).datasets.unstage(args.dataset, args.image)


def _cmd_dataset_list(args: argparse.Namespace) -> Any:
    return Project(args.data_dir).datasets.list()


def _cmd_dataset_status(args: argparse.Namespace) -> Any:
    return Project(args.data_dir).datasets.status(args.name)


def _cmd_dataset_prepare(args: argparse.Namespace) -> Any:
    return Project(args.data_dir).datasets.prepare(
        args.dataset,
        _prepare_config(args, default_size=DEFAULT_PREPARE_SIZE),
    )


def _cmd_models_list(args: argparse.Namespace) -> Any:
    return Project(args.data_dir).models.list()


def _cmd_models_search(args: argparse.Namespace) -> Any:
    return Project(args.data_dir).models.search(
        args.query,
        task=args.task,
        limit=args.limit,
    )


def _cmd_models_fetch(args: argparse.Namespace) -> Any:
    return Project(args.data_dir).models.fetch(args.repo_id, name=args.name)


def _cmd_models_remove(args: argparse.Namespace) -> Any:
    _confirm_or_error(args.yes, f"Remove model '{args.name}'?")
    return Project(args.data_dir).models.remove(args.name)


def _cmd_train(args: argparse.Namespace) -> Any:
    prepare_config = _prepare_config(args, default_size=args.resolution)
    train_values = _train_values(args)
    return Project(args.data_dir).train(
        TrainingSpec(
            dataset=args.dataset,
            model=args.model,
            backend=args.backend,
            resolution=args.resolution,
            rank=train_values.rank,
            steps=train_values.steps,
            learning_rate=train_values.learning_rate,
            batch_size=train_values.batch_size,
            gradient_accumulation=train_values.gradient_accumulation,
            mixed_precision=args.mixed_precision,
            run_name=args.run_name,
            no_prepare=args.no_prepare,
            dry_run=args.dry_run,
            prepare=prepare_config,
        )
    )


def _train_values(args: argparse.Namespace) -> TrainPreset:
    preset = TRAIN_PRESETS.get(args.preset) if args.preset is not None else None
    return TrainPreset(
        description="" if preset is None else preset.description,
        rank=_value_or_default(args.rank, preset.rank if preset else DEFAULT_RANK),
        steps=_value_or_default(args.steps, preset.steps if preset else DEFAULT_STEPS),
        learning_rate=_value_or_default(
            args.learning_rate,
            preset.learning_rate if preset else DEFAULT_LEARNING_RATE,
        ),
        batch_size=_value_or_default(
            args.batch_size,
            preset.batch_size if preset else DEFAULT_BATCH_SIZE,
        ),
        gradient_accumulation=_value_or_default(
            args.gradient_accumulation,
            preset.gradient_accumulation if preset else DEFAULT_GRADIENT_ACCUMULATION,
        ),
    )


def _value_or_default(value: Any, default: Any) -> Any:
    if value is None:
        return default
    return value


def _prepare_config(args: argparse.Namespace, *, default_size: int) -> PrepareConfig:
    if getattr(args, "size", None) is not None and (
        args.width is not None or args.height is not None
    ):
        raise LorakitError("--size cannot be combined with --width or --height")
    if args.mode == "copy":
        if (
            getattr(args, "size", None) is not None
            or args.width is not None
            or args.height is not None
        ):
            raise LorakitError("copy mode does not accept --size, --width, or --height")
        return PrepareConfig(
            mode=args.mode,
            width=None,
            height=None,
            image_format=args.image_format,
            trigger=args.trigger,
        )
    if getattr(args, "size", None) is not None:
        width = args.size
        height = args.size
    elif args.width is None and args.height is None:
        width = default_size
        height = default_size
    else:
        width = args.width
        height = args.height
    return PrepareConfig(
        mode=args.mode,
        width=width,
        height=height,
        image_format=args.image_format,
        trigger=args.trigger,
    )


def _confirm_or_error(yes: bool, prompt: str) -> None:
    if yes:
        return
    if not sys.stdin.isatty():
        raise LorakitError("destructive command requires --yes in non-interactive mode")
    answer = input(f"{prompt} [y/N] ")
    if answer.lower() not in {"y", "yes"}:
        raise LorakitError("operation cancelled")


def _emit(value: Any, output: str, args: argparse.Namespace) -> None:
    if output == "json":
        print(json.dumps(_jsonable(value), indent=2, sort_keys=True))
    elif output == "jsonl":
        values = value if isinstance(value, list) else [value]
        for row in values:
            print(json.dumps(_jsonable(row), sort_keys=True))
    else:
        print(_text(value, args))


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _text(value: Any, args: argparse.Namespace) -> str:
    if isinstance(value, CleanResult):
        return _text_clean_result(value)
    if isinstance(value, DatasetCreated):
        return f"Created dataset {value.name} at {value.path}"
    if isinstance(value, DatasetStatus):
        return _text_dataset_status(value)
    if isinstance(value, ImportResult):
        return _text_import_result(value)
    if isinstance(value, TrainingResult):
        return json.dumps(_jsonable(value), indent=2, sort_keys=True)
    if isinstance(value, list):
        if not value:
            return _text_empty_list(args)
        if all(isinstance(item, Candidate) for item in value):
            return _text_candidates(value)
        if all(isinstance(item, DatasetSummary) for item in value):
            return _text_dataset_summaries(value)
        if all(isinstance(item, ModelInfo) for item in value):
            return _text_models(value)
        if all(isinstance(item, TagResult) for item in value):
            return _text_tag_results(value)
        return "\n".join(_text(item, args) for item in value)
    if is_dataclass(value):
        data = _jsonable(value)
        return json.dumps(data, sort_keys=True)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return json.dumps(_jsonable(value), indent=2, sort_keys=True)
    return str(value)


def _text_empty_list(args: argparse.Namespace) -> str:
    if args.command == "candidates" and args.candidate_command == "list":
        return _text_candidates([])
    if args.command == "dataset" and args.dataset_command == "list":
        return _text_dataset_summaries([])
    if args.command == "models" and args.models_command == "list":
        return _text_models([])
    return ""


def _text_candidates(candidates: list[Candidate]) -> str:
    valid = [candidate for candidate in candidates if not candidate.is_broken]
    broken = [candidate for candidate in candidates if candidate.is_broken]
    visible = valid if len(broken) == 0 else candidates
    name_width = max([len("# NAME"), *(len(candidate.stem) for candidate in visible)])
    image_width = max(
        [
            len("IMAGE"),
            *(len(str(candidate.image)) for candidate in visible if candidate.image is not None),
        ]
    )
    lines = [
        f"# Candidates: {len(candidates)}",
        f"# Valid:      {len(valid)}",
    ]
    if broken:
        lines.append(f"# Broken:     {len(broken)}")
    lines.extend(
        [
            "#",
            f"# {'NAME':<{name_width}}  {'IMAGE':<{image_width}}  HAS_JSON",
        ]
    )
    for candidate in visible:
        image = "" if candidate.image is None else str(candidate.image)
        has_json = "yes" if candidate.metadata is not None else "no"
        lines.append(f"{candidate.stem:<{name_width}}  {image:<{image_width}}  {has_json}")
    return "\n".join(lines)


def _text_dataset_summaries(summaries: list[DatasetSummary]) -> str:
    name_width = max([len("NAME"), *(len(summary.name) for summary in summaries)])
    lines = [f"{'NAME':<{name_width}}  STAGED IMAGES"]
    lines.extend(f"{summary.name:<{name_width}}  {summary.staged_images}" for summary in summaries)
    return "\n".join(lines)


def _text_dataset_status(status: DatasetStatus) -> str:
    prepared = "yes" if status.prepared else "no"
    return "\n".join(
        [
            f"Dataset: {status.name}",
            f"Staged images:    {status.staged_images}",
            f"Overrides:        {status.overrides}",
            f"Missing metadata: {status.missing_metadata}",
            f"Prepared:         {prepared}",
            f"Artifacts:        {status.artifacts}",
        ]
    )


def _text_models(models: list[ModelInfo]) -> str:
    name_width = max([len("NAME"), *(len(model.name) for model in models)])
    type_width = max([len("TYPE"), *(len(model.type) for model in models)])
    format_width = max([len("FORMAT"), *(len(model.format) for model in models)])
    lines = [
        f"{'NAME':<{name_width}}  {'TYPE':<{type_width}}  {'FORMAT':<{format_width}}  PATH"
    ]
    lines.extend(
        f"{model.name:<{name_width}}  {model.type:<{type_width}}  "
        f"{model.format:<{format_width}}  {model.path}"
        for model in models
    )
    return "\n".join(lines)


def _text_clean_result(result: CleanResult) -> str:
    if not result.orphans:
        return "No orphans found."
    action = "Deleted" if result.deleted else "Would delete"
    lines = [f"{action}: {len(result.orphans)}"]
    lines.extend(f"{orphan.path}\t{orphan.reason}" for orphan in result.orphans)
    return "\n".join(lines)


def _text_import_result(result: ImportResult) -> str:
    return "\n".join(
        [
            f"Imported: {len(result.imported)}",
            f"Skipped:  {len(result.skipped)}",
        ]
    )


def _text_tag_results(results: list[TagResult]) -> str:
    tagged = [result for result in results if not result.skipped]
    skipped = [result for result in results if result.skipped]
    lines = [
        f"Tagged:  {len(tagged)}",
        f"Skipped: {len(skipped)}",
    ]
    for result in tagged:
        tag_text = ", ".join(result.added_tags)
        lines.append(f"{result.stem}\t+{len(result.added_tags)}\t{tag_text}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
