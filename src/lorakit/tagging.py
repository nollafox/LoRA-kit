"""Candidate image auto-tagging."""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from PIL import Image, ImageOps

from lorakit.errors import LorakitError
from lorakit.manifest import load_metadata, normalize_tags, write_json


SMILINGWOLF_REPO_ID = "SmilingWolf/wd-vit-tagger-v3"
SMILINGWOLF_MODEL_FILE = "model.onnx"
SMILINGWOLF_TAGS_FILE = "selected_tags.csv"
SMILINGWOLF_THRESHOLD = 0.2614
SMILINGWOLF_GENERAL_CATEGORY = 0
SMILINGWOLF_CHARACTER_CATEGORY = 4
SMILINGWOLF_TAG_CATEGORIES = {
    SMILINGWOLF_GENERAL_CATEGORY,
    SMILINGWOLF_CHARACTER_CATEGORY,
}

JOYCAPTION_REPO_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"
JOYCAPTION_PROMPT = (
    "Write concise natural-language training tags for this image as a comma-separated "
    "list. Describe visible subject, pose, clothing, setting, mood, and composition. "
    "Return only the comma-separated tag list."
)


class ImageTagger(Protocol):
    def tags_for(self, image_path: Path) -> list[str]:
        """Return tags for one image.

        Raises:
            LorakitError: if the model or image cannot be processed.
        """


@dataclass(frozen=True)
class CompositeTagger:
    primary: ImageTagger
    natural: bool = False

    def tags_for(self, image_path: Path) -> list[str]:
        tags: list[str] = []
        seen: set[str] = set()
        for tag in self.primary.tags_for(image_path):
            _append_tag(tags, seen, tag)
        if self.natural:
            for tag in JoyCaptionTagger().tags_for(image_path):
                _append_tag(tags, seen, tag)
        return tags


class SmilingWolfTagger:
    def __init__(self) -> None:
        try:
            self._model_path = hf_hub_download(
                repo_id=SMILINGWOLF_REPO_ID,
                filename=SMILINGWOLF_MODEL_FILE,
            )
            self._tags_path = hf_hub_download(
                repo_id=SMILINGWOLF_REPO_ID,
                filename=SMILINGWOLF_TAGS_FILE,
            )
        except (HfHubHTTPError, LocalEntryNotFoundError) as error:
            raise LorakitError(
                "Could not download SmilingWolf/wd-vit-tagger-v3 files from Hugging "
                "Face. Check your network connection or pre-populate the Hugging Face cache."
            ) from error
        self._session = _onnx_session(self._model_path)
        self._input = self._session.get_inputs()[0]
        self._tags = _load_smilingwolf_tags(Path(self._tags_path))

    def tags_for(self, image_path: Path) -> list[str]:
        image_size = _onnx_image_size(self._input.shape)
        batch = _smilingwolf_image_batch(image_path, image_size)
        outputs = self._session.run(None, {self._input.name: batch})
        if len(outputs) == 0:
            raise LorakitError(f"SmilingWolf tagger returned no outputs for: {image_path}")
        scores = np.asarray(outputs[0][0], dtype=np.float32)
        tags: list[str] = []
        for tag, score in zip(self._tags, scores, strict=False):
            if tag.category in SMILINGWOLF_TAG_CATEGORIES and score >= SMILINGWOLF_THRESHOLD:
                tags.append(tag.name)
        return tags


class JoyCaptionTagger:
    def __init__(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except ImportError as error:
            raise LorakitError(
                "Natural caption tagging requires torch and transformers to be installed"
            ) from error
        self._torch = torch
        try:
            self._processor = AutoProcessor.from_pretrained(JOYCAPTION_REPO_ID)
            self._model = LlavaForConditionalGeneration.from_pretrained(
                JOYCAPTION_REPO_ID,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
        except OSError as error:
            raise LorakitError(
                "Could not load fancyfeast/llama-joycaption-beta-one-hf-llava. "
                "Check your network connection, Hugging Face access, or local cache."
            ) from error

    def tags_for(self, image_path: Path) -> list[str]:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            conversation = [
                {
                    "role": "system",
                    "content": "You are a helpful image captioner.",
                },
                {
                    "role": "user",
                    "content": JOYCAPTION_PROMPT,
                },
            ]
            prompt = self._processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._processor(text=[prompt], images=[image], return_tensors="pt").to(
                self._model.device
            )
        if "pixel_values" in inputs and self._model.dtype.is_floating_point:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)
        with self._torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=160)[0]
        output = output[inputs["input_ids"].shape[1] :]
        text = self._processor.tokenizer.decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return _parse_tag_text(text)


@dataclass(frozen=True)
class SmilingWolfTag:
    name: str
    category: int


def build_tagger(*, natural: bool = False) -> ImageTagger:
    return CompositeTagger(primary=SmilingWolfTagger(), natural=natural)


def merge_tags(metadata_path: Path, image_tags: list[str]) -> tuple[list[str], list[str]]:
    if metadata_path.exists():
        metadata = load_metadata(metadata_path)
    else:
        metadata = {"metadata": {"source": "lorakit"}}
    existing = normalize_tags(metadata.get("tags", []), metadata_path)
    merged: list[str] = []
    seen: set[str] = set()
    for tag in existing:
        _append_tag(merged, seen, tag)
    added: list[str] = []
    for tag in image_tags:
        if tag not in seen:
            _append_tag(merged, seen, tag)
            added.append(tag)
    metadata["tags"] = merged
    write_json(metadata_path, metadata)
    return merged, added


def metadata_has_tags(metadata_path: Path) -> bool:
    if not metadata_path.exists():
        return False
    metadata = load_metadata(metadata_path)
    if "tags" not in metadata:
        return False
    return len(normalize_tags(metadata["tags"], metadata_path)) > 0


def _onnx_session(model_path: str) -> object:
    try:
        import onnxruntime
    except ImportError as error:
        raise LorakitError(
            "SmilingWolf tagging requires onnxruntime >= 1.17.0. "
            "Install it with: python -m pip install 'onnxruntime>=1.17.0'"
        ) from error
    return onnxruntime.InferenceSession(model_path)


def _onnx_image_size(shape: list[object]) -> int:
    dimensions = [dimension for dimension in shape if isinstance(dimension, int)]
    square_dimensions = [dimension for dimension in dimensions if dimension > 3]
    if len(square_dimensions) == 0:
        raise LorakitError(f"Cannot derive SmilingWolf input image size from shape: {shape}")
    return square_dimensions[0]


def _smilingwolf_image_batch(image_path: Path, image_size: int) -> np.ndarray:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = _square_image(image)
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def _square_image(image: Image.Image) -> Image.Image:
    side = max(image.width, image.height)
    background = Image.new("RGB", (side, side), (255, 255, 255))
    left = (side - image.width) // 2
    top = (side - image.height) // 2
    background.paste(image, (left, top))
    return background


def _load_smilingwolf_tags(path: Path) -> list[SmilingWolfTag]:
    tags: list[SmilingWolfTag] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"name", "category"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            joined = ", ".join(sorted(missing))
            raise LorakitError(f"SmilingWolf tag CSV is missing columns: {joined}")
        for row in reader:
            tags.append(SmilingWolfTag(name=row["name"], category=int(row["category"])))
    return tags


def _parse_tag_text(text: str) -> list[str]:
    cleaned = re.sub(r"^[^:]{0,40}:\s*", "", text.strip())
    tags = re.split(r"[,;\n]+", cleaned)
    parsed: list[str] = []
    seen: set[str] = set()
    for raw_tag in tags:
        tag = raw_tag.strip().strip(".").strip()
        if tag:
            _append_tag(parsed, seen, tag)
    return parsed


def _append_tag(tags: list[str], seen: set[str], tag: str) -> None:
    if tag in seen:
        return
    tags.append(tag)
    seen.add(tag)
