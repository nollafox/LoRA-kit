"""Readers and writers for lorakit metadata and prepared manifests."""

import json
from pathlib import Path
from typing import Any

from lorakit.errors import MissingMetadata


MANIFEST_NAME = "lorakit-manifest.jsonl"
TAG_CATEGORY_ORDER = (
    "general",
    "species",
    "character",
    "copyright",
    "artist",
    "meta",
    "lore",
    "invalid",
    "contributor",
)


def load_metadata(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as error:
        raise MissingMetadata(f"Metadata is invalid JSON: {path}") from error
    if not isinstance(data, dict):
        raise MissingMetadata(f"Metadata must be a JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped == "":
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise MissingMetadata(
                    f"Manifest entry {line_number} is invalid JSON: {path}"
                ) from error
            if not isinstance(row, dict):
                raise MissingMetadata(
                    f"Manifest entry {line_number} must be a JSON object: {path}"
                )
            rows.append(row)
    return rows


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def build_manifest_entry(
    image_path: Path,
    metadata_path: Path,
    prepared_root: Path,
    trigger: str = "",
    prompt_type: str = "all",
) -> dict[str, Any]:
    metadata = load_metadata(metadata_path)
    if "tags" not in metadata:
        raise MissingMetadata(f"Metadata is missing required 'tags': {metadata_path}")
    raw_tags = normalize_tags(metadata["tags"], metadata_path)
    
    if prompt_type == "tags":
        tags = raw_tags
    elif prompt_type in {"natural", "caption"}:
        tags = [_naturalize_tag(tag) for tag in raw_tags]
    else:  # "all"
        tags = expand_underscore_tags(raw_tags)

    relative_image = image_path.relative_to(prepared_root).as_posix()
    return {
        "image": relative_image,
        "caption": build_prompt(
            tags=raw_tags,
            caption=metadata.get("caption"),
            trigger=trigger,
            prompt_type=prompt_type,
        ),
        "tags": tags,
    }


def build_prompt(
    *,
    tags: list[str],
    caption: Any,
    trigger: str = "",
    prompt_type: str = "all",
) -> str:
    if prompt_type not in {"tags", "natural", "caption", "all"}:
        raise MissingMetadata(f"Unsupported prompt_type: {prompt_type}")
    tag_values = [trigger, *tags] if trigger else list(tags)
    raw = ", ".join(tag_values)
    natural = ", ".join(_naturalize_tag(tag) for tag in tag_values)
    caption_text = caption.strip() if isinstance(caption, str) else ""
    if prompt_type == "tags":
        return raw
    if prompt_type == "natural":
        return natural
    if prompt_type == "caption":
        if caption_text:
            return _caption_with_trigger(caption_text, trigger)
        return natural
    parts = [raw, natural]
    if caption_text:
        parts.append(_caption_with_trigger(caption_text, trigger))
    return ". ".join(part for part in parts if part)


def _naturalize_tag(tag: str) -> str:
    return tag.replace("(", "").replace(")", "").replace("_", " ")


def _caption_with_trigger(caption: str, trigger: str) -> str:
    if not trigger:
        return caption
    return f"{trigger}. {caption}"


def normalize_tags(tags_value: Any, metadata_path: Path) -> list[str]:
    if isinstance(tags_value, list):
        if not all(isinstance(tag, str) for tag in tags_value):
            raise MissingMetadata(
                f"Metadata 'tags' must contain only strings: {metadata_path}"
            )
        return list(tags_value)
    if isinstance(tags_value, dict):
        return _flatten_tag_categories(tags_value, metadata_path)
    raise MissingMetadata(
        f"Metadata 'tags' must be a list of strings or category mapping: {metadata_path}"
    )


def expand_underscore_tags(tags: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        _append_tag(expanded, seen, tag)
        if "_" in tag or "(" in tag:
            _append_tag(expanded, seen, _naturalize_tag(tag))
    return expanded


def _append_tag(tags: list[str], seen: set[str], tag: str) -> None:
    if tag in seen:
        return
    tags.append(tag)
    seen.add(tag)


def _flatten_tag_categories(tags_by_category: dict[Any, Any], metadata_path: Path) -> list[str]:
    if not all(isinstance(category, str) for category in tags_by_category):
        raise MissingMetadata(f"Metadata tag categories must be strings: {metadata_path}")
    ordered_categories = [
        category for category in TAG_CATEGORY_ORDER if category in tags_by_category
    ]
    ordered_categories.extend(
        sorted(category for category in tags_by_category if category not in TAG_CATEGORY_ORDER)
    )

    tags: list[str] = []
    for category in ordered_categories:
        category_tags = tags_by_category[category]
        if not isinstance(category_tags, list) or not all(
            isinstance(tag, str) for tag in category_tags
        ):
            raise MissingMetadata(
                f"Metadata tags for category '{category}' must be a list of strings: "
                f"{metadata_path}"
            )
        tags.extend(category_tags)
    return tags
