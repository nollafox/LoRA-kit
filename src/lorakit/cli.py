"""Command-line interface for lorakit."""

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
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
    TrainingResult,
)


DEFAULT_PREPARE_SIZE = 512


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
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output", choices=["text", "json", "jsonl"], default="text")
    subparsers = parser.add_subparsers(dest="command")

    _add_candidates(subparsers)
    _add_clean(subparsers)
    _add_dataset(subparsers)
    _add_models(subparsers)
    _add_train(subparsers)
    return parser


def _add_candidates(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("candidates")
    commands = parser.add_subparsers(dest="candidate_command", required=True)

    import_parser = commands.add_parser("import")
    import_parser.add_argument("source")
    import_parser.add_argument("importer_args", nargs=argparse.REMAINDER)
    import_parser.add_argument("--overwrite", action="store_true")
    import_parser.set_defaults(handler=_cmd_candidates_import)

    list_parser = commands.add_parser("list")
    list_parser.set_defaults(handler=_cmd_candidates_list)

    show_parser = commands.add_parser("show")
    show_parser.add_argument("stem")
    show_parser.set_defaults(handler=_cmd_candidates_show)


def _add_clean(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("clean")
    parser.add_argument("--apply", action="store_true")
    parser.set_defaults(handler=_cmd_clean)


def _add_dataset(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("dataset")
    commands = parser.add_subparsers(dest="dataset_command", required=True)

    create_parser = commands.add_parser("create")
    create_parser.add_argument("name")
    create_parser.set_defaults(handler=_cmd_dataset_create)

    delete_parser = commands.add_parser("delete")
    delete_parser.add_argument("name")
    delete_parser.add_argument("--yes", action="store_true")
    delete_parser.set_defaults(handler=_cmd_dataset_delete)

    rename_parser = commands.add_parser("rename")
    rename_parser.add_argument("old")
    rename_parser.add_argument("new")
    rename_parser.set_defaults(handler=_cmd_dataset_rename)

    stage_parser = commands.add_parser("stage")
    stage_parser.add_argument("dataset")
    stage_parser.add_argument("image", nargs="?")
    stage_parser.add_argument("--symlink", action="store_true")
    stage_parser.add_argument("--all", action="store_true")
    stage_parser.set_defaults(handler=_cmd_dataset_stage)

    unstage_parser = commands.add_parser("unstage")
    unstage_parser.add_argument("dataset")
    unstage_parser.add_argument("image")
    unstage_parser.set_defaults(handler=_cmd_dataset_unstage)

    list_parser = commands.add_parser("list")
    list_parser.set_defaults(handler=_cmd_dataset_list)

    status_parser = commands.add_parser("status")
    status_parser.add_argument("name")
    status_parser.set_defaults(handler=_cmd_dataset_status)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("dataset")
    _add_prepare_options(prepare_parser)
    prepare_parser.set_defaults(handler=_cmd_dataset_prepare)


def _add_models(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("models")
    commands = parser.add_subparsers(dest="models_command", required=True)

    list_parser = commands.add_parser("list")
    list_parser.set_defaults(handler=_cmd_models_list)

    search_parser = commands.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--task", default="text-to-image")
    search_parser.add_argument("--limit", type=int, default=20)
    search_parser.set_defaults(handler=_cmd_models_search)

    fetch_parser = commands.add_parser("fetch")
    fetch_parser.add_argument("repo_id")
    fetch_parser.add_argument("--name")
    fetch_parser.set_defaults(handler=_cmd_models_fetch)

    remove_parser = commands.add_parser("remove")
    remove_parser.add_argument("name")
    remove_parser.add_argument("--yes", action="store_true")
    remove_parser.set_defaults(handler=_cmd_models_remove)


def _add_train(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train")
    parser.add_argument("dataset")
    parser.add_argument("--model", default="sd15")
    parser.add_argument("--backend", default="diffusers")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--mixed-precision", default="fp16")
    parser.add_argument("--run-name")
    parser.add_argument("--no-prepare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    _add_prepare_options(parser, include_size=True)
    parser.set_defaults(handler=_cmd_train)


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
    overwrite = args.overwrite
    if "--overwrite" in importer_args:
        importer_args.remove("--overwrite")
        overwrite = True
    return Project(args.data_dir).candidates.import_from(
        args.source,
        importer_args,
        overwrite=overwrite,
    )


def _cmd_candidates_list(args: argparse.Namespace) -> Any:
    return Project(args.data_dir).candidates.list()


def _cmd_candidates_show(args: argparse.Namespace) -> Any:
    return Project(args.data_dir).candidates.show(args.stem)


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
    return Project(args.data_dir).train(
        TrainingSpec(
            dataset=args.dataset,
            model=args.model,
            backend=args.backend,
            resolution=args.resolution,
            rank=args.rank,
            steps=args.steps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            mixed_precision=args.mixed_precision,
            run_name=args.run_name,
            no_prepare=args.no_prepare,
            dry_run=args.dry_run,
            prepare=prepare_config,
        )
    )


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


if __name__ == "__main__":
    raise SystemExit(main())
