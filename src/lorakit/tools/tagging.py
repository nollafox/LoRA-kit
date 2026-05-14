"""Candidate image auto-tagging."""

import csv
import gc
import json
import re
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from PIL import Image, ImageOps
from tqdm.auto import tqdm

import lorakit.models as models
from lorakit.errors import LorakitError
from lorakit.manifest import load_metadata, normalize_tags, write_json


SMILINGWOLF_THRESHOLD = 0.2614
PIPELINE_SMILINGWOLF_THRESHOLD = 0.35
PIPELINE_SMILINGWOLF_CHARACTER_THRESHOLD = 0.85
SMILINGWOLF_GENERAL_CATEGORY = 0
SMILINGWOLF_CHARACTER_CATEGORY = 4
SMILINGWOLF_TAG_CATEGORIES = {
    SMILINGWOLF_GENERAL_CATEGORY,
    SMILINGWOLF_CHARACTER_CATEGORY,
}

FLORENCE_NATURAL_LANGUAGE_PROMPT = "<MORE_DETAILED_CAPTION>"
RAM_PLUS_IMAGE_SIZE = 384
QWEN_MIN_PIXELS = 256 * 28 * 28
QWEN_MAX_PIXELS = 1280 * 28 * 28
MAX_PIPELINE_CANDIDATE_TAGS = 160
MAX_PIPELINE_REJECTED_TAGS = 40
PIPELINE_CAPTION_BACKEND = "Qwen2.5-VL-7B-Instruct"
PIPELINE_EDITOR_BACKEND = "Qwen2.5-VL-7B-Instruct"
PIPELINE_FINAL_TAG_KEYS = (
    "final_tags",
    "objects",
    "scene",
    "medium",
    "style",
    "composition",
    "attributes",
    "actions",
    "booru",
    "characters",
    "abstract_tags",
)
QWEN_SEMANTIC_PROMPT = """
Analyze this image for dataset tagging.

Return STRICT JSON only. No markdown. No comments.

Schema:
{
  "caption": "one concise sentence",
  "content_type": "photo | anime | illustration | painting | digital_art | screenshot | document | mixed | unknown",
  "medium": ["short lowercase tags"],
  "scene": ["short lowercase tags"],
  "objects": ["short lowercase tags"],
  "style": ["short lowercase tags"],
  "composition": ["short lowercase tags"],
  "attributes": ["short lowercase tags"],
  "actions": ["short lowercase tags"],
  "visible_text": ["exact visible text snippets, if any"],
  "abstract_tags": ["mood/theme tags only if strongly supported"],
  "uncertain": ["tags you are unsure about"]
}

Rules:
- Only include visually supported tags.
- Use short noun phrases.
- Prefer lowercase.
- Do not identify real people.
- Do not invent character names.
- For artwork, include medium/style/composition when visible.
""".strip()
CAPTION_EDITOR_SYSTEM_PROMPT = """
You are a caption editor for image-generation LoRA training datasets.

You receive trusted source tags and a draft caption. The trusted source tags are ground truth. The draft caption is only a rough visual description and may contain incorrect species, anatomy, clothing, pose, setting, or subject details.

Your job is to rewrite the draft into one accurate natural-language caption for training.

Rules:
- Output natural language only.
- Do not output booru tags, comma-separated tags, underscore tokens, JSON, markdown, explanations, or notes.
- Treat the trusted source tags as ground truth.
- Every trusted tag that describes visible content should be represented, either directly or by an equivalent natural phrase.
- Convert trusted tags into fluent natural language.
- Preserve all visible adult, explicit, anatomy, pose, interaction, clothing, species, body type, and setting details from the trusted tags.
- Adult and explicit trusted tags are not optional. If a trusted tag describes visible anatomy, nudity, sexual activity, body contact, pose, clothing state, or interaction, it must be represented in neutral natural language unless another trusted tag directly contradicts it.
- Do not sanitize, euphemize, moralize, or omit adult/explicit visual details.
- Keep the wording factual, clinical, and descriptive rather than erotic or literary.
- Correct the draft caption whenever it conflicts with the trusted tags.
- If the draft invents a species, object, clothing item, setting, or character detail not supported by the tags, replace it with the trusted tag equivalent or with a generic phrase.
- Preserve useful visual details from the draft caption when they do not conflict with the tags, including medium, lighting, rendering style, camera angle, composition, mood, color palette, and background.
- Ignore non-visual metadata tags such as score, source, artist name, rating labels, file quality labels, username, website, watermark, signature, and resolution unless they describe visible content that should be learned.
- Do not invent new concrete details.
- Write one or two complete sentences.
- Output only the corrected caption.
""".strip()
CAPTION_EDITOR_TAG_LIMIT = 500
CAPTION_EDITOR_CHAR_LIMIT = 4000
CAPTION_PRESETS = ("natural_language", "image_tags")
DANBOORU_CATEGORY_NAMES = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "meta",
    9: "rating",
}
TAG_ALIASES = {
    "automobile": "car",
    "auto": "car",
    "motorcar": "car",
    "bike": "bicycle",
    "cycle": "bicycle",
    "cellphone": "mobile phone",
    "cell phone": "mobile phone",
    "smartphone": "mobile phone",
    "tv": "television",
    "sofa": "couch",
    "settee": "couch",
    "kitty": "cat",
    "kitten": "cat",
    "puppy": "dog",
    "canine": "dog",
    "human": "person",
    "people": "person",
    "man": "male-presenting person",
    "woman": "female-presenting person",
    "boy": "male-presenting child",
    "girl": "female-presenting child",
    "1girl": "female-presenting person",
    "1boy": "male-presenting person",
    "2girls": "multiple female-presenting people",
    "2boys": "multiple male-presenting people",
    "multiple girls": "multiple female-presenting people",
    "multiple boys": "multiple male-presenting people",
    "solo": "solo subject",
}


@dataclass(frozen=True)
class TagRequest:
    image_path: Path
    context_tags: list[str]


@dataclass(frozen=True)
class TaggingResult:
    tags: list[str]
    caption: str | None = None
    draft_caption: str | None = None
    caption_backend: str | None = None
    editor_backend: str | None = None
    sidecar: dict[str, Any] | None = None


@dataclass(frozen=True)
class CandidateTag:
    raw: str
    tag: str
    source: str
    namespace: str
    confidence: float | None = None
    authority: float = 0.5
    score: float = 0.5


class ImageTagger(Protocol):
    def tags_for(
        self,
        image_path: Path,
        *,
        context_tags: list[str] | None = None,
    ) -> list[str]:
        """Return tags for one image.

        Raises:
            LorakitError: if the model or image cannot be processed.
        """


@dataclass(frozen=True)
class CompositeTagger:
    taggers: tuple[ImageTagger, ...]

    def results_for_many(self, requests: list[TagRequest]) -> list[TaggingResult]:
        tags_by_request: list[list[str]] = [[] for _ in requests]
        seen_by_request: list[set[str]] = [set() for _ in requests]
        captions: list[str | None] = [None for _ in requests]
        draft_captions: list[str | None] = [None for _ in requests]
        caption_backends: list[str | None] = [None for _ in requests]
        editor_backends: list[str | None] = [None for _ in requests]
        for tagger in self.taggers:
            if hasattr(tagger, "results_for_many"):
                tag_results = tagger.results_for_many(requests)
            elif hasattr(tagger, "tags_for_many"):
                tag_results = [
                    TaggingResult(tags=tags) for tags in tagger.tags_for_many(requests)
                ]
            else:
                tag_results = [
                    TaggingResult(
                        tags=tagger.tags_for(
                            request.image_path,
                            context_tags=request.context_tags,
                        )
                    )
                    for request in requests
                ]
            for index, result in enumerate(tag_results):
                for tag in result.tags:
                    _append_tag(tags_by_request[index], seen_by_request[index], tag)
                if result.caption is not None:
                    captions[index] = result.caption
                    draft_captions[index] = result.draft_caption
                    caption_backends[index] = result.caption_backend
                    editor_backends[index] = result.editor_backend
        return [
            TaggingResult(
                tags=tags_by_request[index],
                caption=captions[index],
                draft_caption=draft_captions[index],
                caption_backend=caption_backends[index],
                editor_backend=editor_backends[index],
            )
            for index in range(len(requests))
        ]

    def tags_for_many(self, requests: list[TagRequest]) -> list[list[str]]:
        return [result.tags for result in self.results_for_many(requests)]

    def tags_for(
        self,
        image_path: Path,
        *,
        context_tags: list[str] | None = None,
    ) -> list[str]:
        return self.results_for_many([TagRequest(image_path, context_tags or [])])[0].tags


class SmilingWolfTagger:
    def __init__(self) -> None:
        try:
            self._model_path = hf_hub_download(
                repo_id=models.SMILINGWOLF_REPO_ID,
                filename=models.SMILINGWOLF_MODEL_FILE,
                cache_dir=models.HF_CACHE_DIR,
            )
            self._tags_path = hf_hub_download(
                repo_id=models.SMILINGWOLF_REPO_ID,
                filename=models.SMILINGWOLF_TAGS_FILE,
                cache_dir=models.HF_CACHE_DIR,
            )
        except (HfHubHTTPError, LocalEntryNotFoundError) as error:
            raise LorakitError(
                "Could not download SmilingWolf/wd-vit-tagger-v3 files from Hugging "
                "Face. Check your network connection or pre-populate the Hugging Face cache."
            ) from error
        self._session = _onnx_session(self._model_path)
        self._input = self._session.get_inputs()[0]
        self._tags = _load_smilingwolf_tags(Path(self._tags_path))

    def tags_for(
        self,
        image_path: Path,
        *,
        context_tags: list[str] | None = None,
    ) -> list[str]:
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


class PipelineTagger:
    def __init__(self) -> None:
        try:
            import torch
        except ImportError as error:
            raise LorakitError("Pipeline captioning requires torch to be installed") from error
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def tags_for(
        self,
        image_path: Path,
        *,
        context_tags: list[str] | None = None,
    ) -> list[str]:
        return self.results_for_many([TagRequest(image_path, context_tags or [])])[0].tags

    def results_for_many(self, requests: list[TagRequest]) -> list[TaggingResult]:
        return self.results_for_many_with_progress(requests, quiet=True)

    def results_for_many_with_progress(
        self,
        requests: list[TagRequest],
        *,
        quiet: bool,
    ) -> list[TaggingResult]:
        return self.results_for_many_iteratively(
            requests,
            quiet=quiet,
            on_result=None,
        )

    def results_for_many_iteratively(
        self,
        requests: list[TagRequest],
        *,
        quiet: bool,
        on_result: Callable[[int, TaggingResult], None] | None,
    ) -> list[TaggingResult]:
        candidate_records = self._ram_stage(requests, quiet=quiet)
        qwen = QwenVLTagJudge(device=self._device)
        try:
            semantic_records = self._semantic_stage(
                qwen=qwen,
                requests=requests,
                candidate_records=candidate_records,
                quiet=quiet,
            )
            if _needs_smilingwolf(semantic_records):
                qwen.unload()
                qwen = None
                _clear_cuda(self._torch)
                self._smilingwolf_stage(
                    requests=requests,
                    candidate_records=candidate_records,
                    semantic_records=semantic_records,
                    quiet=quiet,
                )
            return self._qwen_stage(
                requests,
                candidate_records,
                semantic_records,
                qwen=qwen,
                quiet=quiet,
                on_result=on_result,
            )
        finally:
            if qwen is not None:
                qwen.unload()
            _clear_cuda(self._torch)

    def tags_for_many(self, requests: list[TagRequest]) -> list[list[str]]:
        return [result.tags for result in self.results_for_many(requests)]

    def _ram_stage(
        self,
        requests: list[TagRequest],
        *,
        quiet: bool,
    ) -> list[dict[str, Any]]:
        sources: list[tuple[str, object | None]] = []
        load_errors: list[dict[str, str]] = []
        try:
            ram_tagger, error = _load_pipeline_source(
                "ram++",
                RAMPlusTagger,
                device=self._device,
            )
            if error is not None:
                load_errors.append(error)
            sources.append(("ram++", ram_tagger))
            records: list[dict[str, Any]] = []
            for request in _progress(
                requests,
                desc="RAM++ tag sources",
                quiet=quiet,
                total=len(requests),
            ):
                records.append(_candidate_record(request, sources, load_errors))
            return records
        finally:
            _unload_pipeline_sources(source for _, source in sources)
            _clear_cuda(self._torch)

    def _semantic_stage(
        self,
        *,
        qwen: "QwenVLTagJudge",
        requests: list[TagRequest],
        candidate_records: list[dict[str, Any]],
        quiet: bool,
    ) -> list[dict[str, Any]]:
        semantics: list[dict[str, Any]] = []
        for request, candidate_record in _progress(
            zip(requests, candidate_records, strict=True),
            desc="Qwen semantic",
            quiet=quiet,
            total=len(requests),
            stem=lambda pair: pair[0].image_path.stem,
        ):
            errors = candidate_record.setdefault("errors", [])
            semantics.append(_semantic_extract(qwen, request.image_path, errors))
        return semantics

    def _smilingwolf_stage(
        self,
        *,
        requests: list[TagRequest],
        candidate_records: list[dict[str, Any]],
        semantic_records: list[dict[str, Any]],
        quiet: bool,
    ) -> None:
        work = [
            (request, candidate_record)
            for request, candidate_record, semantic in zip(
                requests,
                candidate_records,
                semantic_records,
                strict=True,
            )
            if not _skip_smilingwolf_for_semantic(semantic)
        ]
        if len(work) == 0:
            return
        sw_tagger, error = _load_pipeline_source(
            "smilingwolf",
            PipelineSmilingWolfTagger,
            device=self._device,
        )
        if error is not None:
            for _, candidate_record in work:
                candidate_record.setdefault("errors", []).append(error)
            return
        try:
            for request, candidate_record in _progress(
                work,
                desc="SmilingWolf art tags",
                quiet=quiet,
                total=len(work),
                stem=lambda pair: pair[0].image_path.stem,
            ):
                try:
                    candidate_record["smilingwolf"] = sw_tagger.tag(request.image_path)
                except Exception as error:
                    candidate_record.setdefault("errors", []).append(
                        _stage_error("smilingwolf", error)
                    )
        finally:
            _unload_pipeline_sources([sw_tagger])
            _clear_cuda(self._torch)

    def _qwen_stage(
        self,
        requests: list[TagRequest],
        candidate_records: list[dict[str, Any]],
        semantic_records: list[dict[str, Any]],
        *,
        qwen: Any | None,
        quiet: bool,
        on_result: Callable[[int, TaggingResult], None] | None = None,
    ) -> list[TaggingResult]:
        active_qwen = qwen
        try:
            if active_qwen is None:
                active_qwen = QwenVLTagJudge(device=self._device)
            results: list[TaggingResult] = []
            for index, (request, candidate_record, semantic) in enumerate(_progress(
                zip(requests, candidate_records, semantic_records, strict=True),
                desc="Qwen prune",
                quiet=quiet,
                total=len(requests),
                stem=lambda pair: pair[0].image_path.stem,
            )):
                result = _qwen_result(active_qwen, request, candidate_record, semantic)
                results.append(result)
                if on_result is not None:
                    on_result(index, result)
            return results
        finally:
            if qwen is None and active_qwen is not None:
                active_qwen.unload()
            _clear_cuda(self._torch)


class RAMPlusTagger:
    def __init__(self, *, device: str) -> None:
        try:
            import torch
        except ImportError as error:
            raise LorakitError(
                "Pipeline tagging requires recognize-anything/RAM++. Install it with: "
                "python -m pip install 'git+https://github.com/xinyu1205/recognize-anything.git'"
            ) from error
        _patch_transformers_for_ram_plus()
        try:
            from ram import get_transform
            from ram import inference_ram as inference
            from ram.models import ram_plus
        except ImportError as error:
            raise LorakitError(
                "Pipeline tagging requires recognize-anything/RAM++. Install it with: "
                "python -m pip install 'git+https://github.com/xinyu1205/recognize-anything.git'"
            ) from error
        self._torch = torch
        self._device = torch.device(device)
        self._transform = get_transform(image_size=RAM_PLUS_IMAGE_SIZE)
        self._inference = inference
        checkpoint = hf_hub_download(
            repo_id=models.RAM_PLUS_REPO_ID,
            filename=models.RAM_PLUS_FILENAME,
            cache_dir=models.HF_CACHE_DIR,
        )
        self._model = ram_plus(
            pretrained=str(checkpoint),
            image_size=RAM_PLUS_IMAGE_SIZE,
            vit="swin_l",
        )
        self._model.eval()
        self._model.to(self._device)

    def tag(self, image_path: Path) -> dict[str, Any]:
        image = _load_rgb_image(image_path)
        tensor = self._transform(image).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            result = self._inference(tensor, self._model)
        english = ""
        chinese = ""
        if isinstance(result, (list, tuple)):
            if len(result) > 0:
                english = str(result[0])
            if len(result) > 1:
                chinese = str(result[1])
        else:
            english = str(result)
        tags = [
            normalize_candidate_tag(tag)
            for tag in re.split(r"\s*\|\s*", english)
            if tag.strip()
        ]
        return {
            "english_raw": english,
            "chinese_raw": chinese,
            "tags": unique_preserve_order(tags),
        }

    def unload(self) -> None:
        self._model = None
        _clear_cuda(self._torch)


class PipelineSmilingWolfTagger:
    def __init__(self, *, device: str) -> None:
        try:
            import timm
            import torch
            from timm.data import create_transform, resolve_data_config
        except ImportError as error:
            raise LorakitError("Pipeline SmilingWolf tagging requires timm and torch") from error
        self._torch = torch
        self._device = torch.device(device)
        self._model = timm.create_model(
            f"hf_hub:{models.DEFAULT_PIPELINE_SMILINGWOLF_MODEL}",
            pretrained=True,
        )
        self._model.eval()
        self._model.to(self._device)
        data_config = resolve_data_config({}, model=self._model)
        self._transform = create_transform(**data_config, is_training=False)
        tags_csv = hf_hub_download(
            repo_id=models.DEFAULT_PIPELINE_SMILINGWOLF_MODEL,
            filename=models.SMILINGWOLF_TAGS_FILE,
            cache_dir=models.HF_CACHE_DIR,
        )
        self._tag_rows = self._load_tag_rows(Path(tags_csv))

    def _load_tag_rows(self, csv_path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = row.get("name") or row.get("tag") or row.get("tag_name")
                if not name:
                    continue
                try:
                    category = int(row.get("category", "-1"))
                except ValueError:
                    category = -1
                rows.append(
                    {
                        "raw": name,
                        "normalized": normalize_candidate_tag(name),
                        "category": category,
                        "category_name": DANBOORU_CATEGORY_NAMES.get(category, "unknown"),
                    }
                )
        return rows

    def tag(self, image_path: Path) -> dict[str, Any]:
        image = _pad_to_square(_load_rgb_image(image_path))
        tensor = self._transform(image).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            output = self._model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        probabilities = self._torch.sigmoid(output).detach().float().cpu().numpy()[0]
        if len(probabilities) != len(self._tag_rows):
            raise LorakitError(
                "SmilingWolf output/tag mismatch: "
                f"{len(probabilities)} probabilities vs {len(self._tag_rows)} tags"
            )
        rating_candidates = []
        general_tags = []
        character_tags = []
        copyright_tags = []
        artist_tags = []
        meta_tags = []
        all_selected = []
        for probability, row in zip(probabilities, self._tag_rows, strict=False):
            confidence = float(probability)
            item = {
                "raw": row["raw"],
                "tag": row["normalized"],
                "confidence": confidence,
                "category": row["category"],
                "category_name": row["category_name"],
            }
            if row["category"] == 9:
                rating_candidates.append(item)
                continue
            threshold = (
                PIPELINE_SMILINGWOLF_CHARACTER_THRESHOLD
                if row["category"] == SMILINGWOLF_CHARACTER_CATEGORY
                else PIPELINE_SMILINGWOLF_THRESHOLD
            )
            if confidence < threshold:
                continue
            all_selected.append(item)
            if row["category"] == SMILINGWOLF_CHARACTER_CATEGORY:
                character_tags.append(item)
            elif row["category"] == 3:
                copyright_tags.append(item)
            elif row["category"] == 1:
                artist_tags.append(item)
            elif row["category"] == 5:
                meta_tags.append(item)
            else:
                general_tags.append(item)
        rating = max(rating_candidates, key=lambda item: item["confidence"], default=None)
        all_selected.sort(key=lambda item: item["confidence"], reverse=True)
        return {
            "rating": rating,
            "general": general_tags,
            "characters": character_tags,
            "copyrights": copyright_tags,
            "artists": artist_tags,
            "meta": meta_tags,
            "selected": all_selected,
        }

    def unload(self) -> None:
        self._model = None
        _clear_cuda(self._torch)


class QwenVLTagJudge:
    def __init__(self, *, device: str, model_id: str = models.DEFAULT_QWEN_VL_MODEL) -> None:
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoModelForImageTextToText, AutoProcessor
            from transformers import BitsAndBytesConfig
        except ImportError as error:
            raise LorakitError(
                "Pipeline captioning requires transformers, bitsandbytes, and qwen-vl-utils"
            ) from error
        self._torch = torch
        self._process_vision_info = process_vision_info
        self._device = device
        self._model_id = model_id
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=QWEN_MIN_PIXELS,
            max_pixels=QWEN_MAX_PIXELS,
            cache_dir=models.HF_CACHE_DIR,
        )
        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
            "cache_dir": models.HF_CACHE_DIR,
        }
        if torch.cuda.is_available() and device != "cpu":
            model_kwargs["dtype"] = torch.float16
            model_kwargs["max_memory"] = _cuda_max_memory(torch)
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["dtype"] = torch.float32
        self._model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        self._model.eval()

    def semantic_extract(self, image_path: Path) -> dict[str, Any]:
        raw = self._chat_image(image_path, QWEN_SEMANTIC_PROMPT, max_new_tokens=512)
        obj = extract_json_object(raw)
        obj["_raw_response"] = raw
        return obj

    def prune_candidates(
        self,
        *,
        image_path: Path,
        semantic: dict[str, Any],
        candidates: list[CandidateTag],
        max_rejected: int,
    ) -> dict[str, Any]:
        prompt = _qwen_prune_prompt(
            semantic=semantic,
            candidates=candidates,
            max_rejected=max_rejected,
        )
        raw = self._chat_image(image_path, prompt, max_new_tokens=900)
        obj = extract_json_object(raw)
        obj["_raw_response"] = raw
        candidate_set = {candidate.tag for candidate in candidates}
        for key in PIPELINE_FINAL_TAG_KEYS:
            if isinstance(obj.get(key), list):
                obj[key] = unique_preserve_order(
                    [
                        normalized
                        for item in obj[key]
                        if (normalized := normalize_candidate_tag(str(item))) in candidate_set
                    ]
                )
        if isinstance(obj.get("visible_text"), list):
            obj["visible_text"] = unique_preserve_order(
                [str(item).strip() for item in obj["visible_text"] if str(item).strip()]
            )
        return obj

    def _chat_image(self, image_path: Path, prompt: str, max_new_tokens: int) -> str:
        image_input = str(image_path.resolve())
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_input},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        target_device = "cuda" if self._torch.cuda.is_available() and self._device != "cpu" else "cpu"
        inputs = inputs.to(target_device)
        with self._torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        return self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def unload(self) -> None:
        self._model = None
        self._processor = None
        _clear_cuda(self._torch)


class FlorencePromptGenTagger:
    def __init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise LorakitError(
                "Natural language captioning requires torch and transformers to be installed"
            ) from error
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._editor: CaptionEditor | None = None
        try:
            self._processor = AutoProcessor.from_pretrained(
                models.FLORENCE_PROMPTGEN_REPO_ID,
                use_fast=False,
                cache_dir=models.HF_CACHE_DIR,
            )
            self._model_class = AutoModelForImageTextToText
        except OSError as error:
            raise LorakitError(
                "Could not load Disty0/Florence-2-large-PromptGen-v2.0. "
                "Check your network connection, Hugging Face access, or local cache."
            ) from error

    def tags_for(
        self,
        image_path: Path,
        *,
        context_tags: list[str] | None = None,
    ) -> list[str]:
        return self.results_for_many([TagRequest(image_path, context_tags or [])])[0].tags

    def results_for_many(self, requests: list[TagRequest]) -> list[TaggingResult]:
        drafts = [self._draft_caption(request.image_path) for request in requests]
        self._release_model()
        results: list[TaggingResult] = []
        for request, draft in zip(requests, drafts, strict=True):
            text = draft
            editor_backend = None
            if request.context_tags:
                text = self._caption_editor().correct_caption(
                    trusted_tags=request.context_tags,
                    draft_caption=draft,
                )
                editor_backend = "Dolphin3.0-Llama3.1-8B"
            results.append(
                TaggingResult(
                    tags=[],
                    caption=text,
                    draft_caption=draft,
                    caption_backend="Florence-2-large-PromptGen",
                    editor_backend=editor_backend,
                )
            )
        return results

    def tags_for_many(self, requests: list[TagRequest]) -> list[list[str]]:
        return [result.tags for result in self.results_for_many(requests)]

    def _draft_caption(self, image_path: Path) -> str:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            inputs = self._processor(
                text=FLORENCE_NATURAL_LANGUAGE_PROMPT,
                images=image,
                return_tensors="pt",
            ).to(self._device)
            image_size = (image.width, image.height)
        with self._torch.no_grad():
            generated_ids = self._model_for_generation().generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=300,
                do_sample=False,
                num_beams=3,
            )
        generated_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]
        parsed = self._processor.post_process_generation(
            generated_text,
            task=FLORENCE_NATURAL_LANGUAGE_PROMPT,
            image_size=image_size,
        )
        text = parsed.get(FLORENCE_NATURAL_LANGUAGE_PROMPT, generated_text)
        if not isinstance(text, str):
            raise LorakitError(
                f"Florence-2 PromptGen returned an unexpected caption for: {image_path}"
            )
        return text

    def _model_for_generation(self) -> object:
        if self._model is None:
            self._model = self._model_class.from_pretrained(
                models.FLORENCE_PROMPTGEN_REPO_ID,
                cache_dir=models.HF_CACHE_DIR,
            )
            self._model.to(self._device)
            self._model.eval()
        return self._model

    def _release_model(self) -> None:
        if self._model is None:
            return
        try:
            self._model.to("cpu")
        except RuntimeError:
            pass
        self._model = None
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _caption_editor(self) -> "CaptionEditor":
        if self._editor is None:
            self._editor = CaptionEditor()
        return self._editor


class CaptionEditor:
    def __init__(self, model_id: str = models.DEFAULT_CAPTION_EDITOR_MODEL) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as error:
            raise LorakitError(
                "Caption editing requires torch, transformers, accelerate, bitsandbytes, "
                "and safetensors to be installed"
            ) from error
        self._torch = torch
        self._model_id = model_id
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=models.HF_CACHE_DIR,
            )
            model_kwargs: dict[str, object] = {
                "device_map": "auto",
                "trust_remote_code": True,
                "cache_dir": models.HF_CACHE_DIR,
            }
            if torch.cuda.is_available():
                model_kwargs["dtype"] = torch.float16
                model_kwargs["max_memory"] = _cuda_max_memory(torch)
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            self._model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            self._model.eval()
        except OSError as error:
            raise LorakitError(
                f"Could not load {model_id}. Check your network connection, Hugging Face "
                "access, or local cache."
            ) from error

    def correct_caption(
        self,
        *,
        trusted_tags: list[str],
        draft_caption: str,
        trigger: str | None = None,
    ) -> str:
        tag_text = ", ".join(_caption_editor_tags(trusted_tags))
        user_prompt = f"""Trusted source tags:
{tag_text}

Draft caption:
{draft_caption}

Rewrite the draft as a natural-language training caption."""
        messages = [
            {"role": "system", "content": CAPTION_EDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        caption = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return clean_caption_output(caption)


@dataclass(frozen=True)
class SmilingWolfTag:
    name: str
    category: int


def build_tagger(*, presets: list[str] | tuple[str, ...] | None = None) -> ImageTagger:
    selected = list(presets) if presets is not None else ["image_tags"]
    if len(selected) == 0:
        raise LorakitError("At least one caption preset is required")

    if "natural_language" in selected:
        unknown = [preset for preset in selected if preset not in CAPTION_PRESETS]
        if unknown:
            joined = ", ".join(CAPTION_PRESETS)
            raise LorakitError(f"Unknown caption preset '{unknown[0]}'. Choose from: {joined}")
        return PipelineTagger()

    taggers: list[ImageTagger] = []
    for preset in selected:
        if preset == "image_tags":
            taggers.append(SmilingWolfTagger())
        else:
            joined = ", ".join(CAPTION_PRESETS)
            raise LorakitError(f"Unknown caption preset '{preset}'. Choose from: {joined}")
    return CompositeTagger(taggers=tuple(taggers))


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


def merge_caption(metadata_path: Path, result: TaggingResult) -> str | None:
    if result.caption is None and result.sidecar is None:
        return None
    if metadata_path.exists():
        metadata = load_metadata(metadata_path)
    else:
        metadata = {"metadata": {"source": "lorakit"}}
    if result.caption is not None:
        metadata["caption"] = result.caption
    metadata_block = metadata.get("metadata", {})
    if not isinstance(metadata_block, dict):
        metadata_block = {}
    if result.caption is not None:
        captioning: dict[str, str] = {
            "draft_caption": result.draft_caption or result.caption,
            "caption": result.caption,
        }
        if result.caption_backend is not None:
            captioning["caption_backend"] = result.caption_backend
        if result.editor_backend is not None:
            captioning["editor_backend"] = result.editor_backend
        metadata_block["captioning"] = captioning
    if result.sidecar is not None:
        metadata_block["tagging"] = result.sidecar
    metadata["metadata"] = metadata_block
    write_json(metadata_path, metadata)
    return result.caption


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


def _caption_editor_tags(tags: list[str]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    used_chars = 0
    for tag in tags:
        cleaned = re.sub(r"\s+", " ", tag).strip()
        if not cleaned or cleaned in seen:
            continue
        extra_chars = len(cleaned) + (2 if selected else 0)
        if selected and (
            len(selected) >= CAPTION_EDITOR_TAG_LIMIT
            or used_chars + extra_chars > CAPTION_EDITOR_CHAR_LIMIT
        ):
            break
        selected.append(cleaned)
        seen.add(cleaned)
        used_chars += extra_chars
    return selected


def clean_caption_output(text: str) -> str:
    cleaned = text.strip()
    for prefix in ("Corrected caption:", "Caption:", "Output:"):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
    cleaned = cleaned.split("\n\n", 1)[0].strip()
    return cleaned.strip('"').strip("'").strip()


def _cuda_max_memory(torch_module: object) -> dict[object, str]:
    device = torch_module.cuda.current_device()
    _, total_bytes = torch_module.cuda.mem_get_info(device)
    gib = 1024**3
    budget_gib = max(1, int(total_bytes * 0.75) // gib)
    return {device: f"{budget_gib}GiB", "cpu": "64GiB"}


def _progress(
    items: Iterable[Any],
    *,
    desc: str,
    quiet: bool,
    total: int,
    stem: Any | None = None,
) -> Iterable[Any]:
    iterator = tqdm(
        items,
        total=total,
        desc=desc,
        unit="image",
        dynamic_ncols=True,
        leave=False,
        disable=quiet or total == 0,
    )
    for item in iterator:
        label = stem(item) if stem is not None else getattr(item, "image_path", item).stem
        iterator.set_postfix_str(str(label), refresh=False)
        yield item


def _load_pipeline_source(
    name: str,
    source_class: Any,
    *,
    device: str,
) -> tuple[object | None, dict[str, str] | None]:
    try:
        return source_class(device=device), None
    except Exception as error:
        return None, _stage_error(f"{name}_load", error)


def _unload_pipeline_sources(sources: Iterable[object | None]) -> None:
    for source in sources:
        if source is None:
            continue
        unload = getattr(source, "unload", None)
        if callable(unload):
            unload()


def _candidate_record(
    request: TagRequest,
    sources: list[tuple[str, object | None]],
    load_errors: list[dict[str, str]],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "ram++": None,
        "smilingwolf": None,
        "errors": list(load_errors),
    }
    for name, source in sources:
        if source is None:
            continue
        try:
            record[name] = source.tag(request.image_path)
        except Exception as error:
            record["errors"].append(_stage_error(name, error))
    return record


def _qwen_result(
    qwen: "QwenVLTagJudge",
    request: TagRequest,
    candidate_record: dict[str, Any],
    semantic: dict[str, Any],
) -> TaggingResult:
    errors = list(candidate_record.get("errors", []))
    raw_candidates = _raw_pipeline_candidates(
        context_tags=request.context_tags,
        candidate_record=candidate_record,
        semantic=semantic,
    )
    merged_candidates = merge_candidates(
        raw_candidates,
        max_candidates=MAX_PIPELINE_CANDIDATE_TAGS,
    )
    final = _prune_candidates(qwen, request.image_path, semantic, merged_candidates, errors)
    caption = _final_caption(final, semantic)
    qwen_final_tags = _string_list(final.get("final_tags"))
    final_tags = _with_required_tags(
        _final_tags_or_model_output_fallback(final, merged_candidates),
        [
            *_semantic_tags_from_result(semantic),
            *_ram_tags_from_candidate_record(candidate_record),
        ],
    )
    sidecar_final = _without_raw_response(final)
    sidecar_final["final_tags"] = final_tags
    sidecar = {
        "semantic": _without_raw_response(semantic),
        "final": sidecar_final,
        "source_tags": build_source_tags(raw_candidates),
        "errors": errors,
    }
    if not qwen_final_tags and final_tags:
        sidecar["tag_fallback"] = "merged_model_outputs"
    return TaggingResult(
        tags=final_tags,
        caption=caption,
        draft_caption=_draft_caption_from_semantic(semantic, caption),
        caption_backend=PIPELINE_CAPTION_BACKEND,
        editor_backend=PIPELINE_EDITOR_BACKEND,
        sidecar=sidecar,
    )


def _semantic_extract(
    qwen: "QwenVLTagJudge",
    image_path: Path,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    try:
        return qwen.semantic_extract(image_path)
    except Exception as error:
        errors.append(_stage_error("qwen_semantic", error))
        return {
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }


def _needs_smilingwolf(semantic_records: list[dict[str, Any]]) -> bool:
    return any(
        not _skip_smilingwolf_for_semantic(semantic)
        for semantic in semantic_records
    )


def _skip_smilingwolf_for_semantic(semantic: dict[str, Any]) -> bool:
    content_type = normalize_candidate_tag(str(semantic.get("content_type", "")))
    return content_type == "photo"


def _raw_pipeline_candidates(
    *,
    context_tags: list[str],
    candidate_record: dict[str, Any],
    semantic: dict[str, Any],
) -> list[CandidateTag]:
    candidates: list[CandidateTag] = []
    candidates.extend(candidates_from_context(context_tags))
    if isinstance(candidate_record.get("ram++"), dict):
        candidates.extend(candidates_from_ram(candidate_record["ram++"]))
    if isinstance(candidate_record.get("smilingwolf"), dict):
        candidates.extend(candidates_from_pipeline_smilingwolf(candidate_record["smilingwolf"]))
    if not semantic.get("error"):
        candidates.extend(candidates_from_qwen_semantic(semantic))
    return candidates


def _prune_candidates(
    qwen: "QwenVLTagJudge",
    image_path: Path,
    semantic: dict[str, Any],
    merged_candidates: list[CandidateTag],
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    try:
        return qwen.prune_candidates(
            image_path=image_path,
            semantic=semantic,
            candidates=merged_candidates,
            max_rejected=MAX_PIPELINE_REJECTED_TAGS,
        )
    except Exception as error:
        errors.append(_stage_error("qwen_prune", error))
        return {
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "final_tags": [candidate.tag for candidate in merged_candidates],
            "note": "Qwen prune failed; using unpruned merged candidates.",
        }


def _final_caption(final: dict[str, Any], semantic: dict[str, Any]) -> str | None:
    caption = final.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        caption = semantic.get("caption") if isinstance(semantic, dict) else None
    return caption.strip() if isinstance(caption, str) and caption.strip() else None


def _draft_caption_from_semantic(
    semantic: dict[str, Any],
    fallback: str | None,
) -> str | None:
    caption = semantic.get("caption")
    return caption if isinstance(caption, str) else fallback


def _without_raw_response(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key != "_raw_response"}


def _final_tags_or_model_output_fallback(
    final: dict[str, Any],
    merged_candidates: list[CandidateTag],
) -> list[str]:
    final_tags = _string_list(final.get("final_tags"))
    if final_tags:
        return final_tags
    return [candidate.tag for candidate in merged_candidates]


def _ram_tags_from_candidate_record(candidate_record: dict[str, Any]) -> list[str]:
    ram_result = candidate_record.get("ram++")
    if not isinstance(ram_result, dict):
        return []
    return _string_list(ram_result.get("tags"))


def _content_type_tags(semantic: dict[str, Any]) -> list[str]:
    content_type = semantic.get("content_type")
    if not isinstance(content_type, str):
        return []
    tag = normalize_candidate_tag(content_type)
    if not tag or tag == "unknown":
        return []
    return [tag]


def _semantic_tags_from_result(semantic: dict[str, Any]) -> list[str]:
    return unique_preserve_order(
        [
            candidate.tag
            for candidate in candidates_from_qwen_semantic(semantic)
            if candidate.tag
        ]
    )


def _with_required_tags(tags: list[str], required_tags: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for tag in [*tags, *required_tags]:
        _append_tag(output, seen, tag)
    return output


def _qwen_prune_prompt(
    *,
    semantic: dict[str, Any],
    candidates: list[CandidateTag],
    max_rejected: int,
) -> str:
    candidate_payload = [
        {
            "tag": candidate.tag,
            "raw": candidate.raw,
            "source": candidate.source,
            "namespace": candidate.namespace,
            "score": round(candidate.score, 4),
            "confidence": (
                None if candidate.confidence is None else round(candidate.confidence, 4)
            ),
        }
        for candidate in candidates
    ]
    return f"""
You are pruning candidate tags for dataset tagging.

You are given:
1. An image.
2. A semantic analysis.
3. A candidate tag list produced by other models.

Task:
- Remove candidate tags that are not visibly supported by the image.
- Keep candidate tags that are clearly or reasonably visually supported.
- Do NOT add new tags that are not in the candidate list.
- Do NOT identify real people.
- Preserve useful booru/anime tags when they are visibly supported.
- Prefer useful dataset-search tags over tiny irrelevant details.

Return STRICT JSON only. No markdown.

Schema:
{{
  "caption": "one concise sentence",
  "content_type": "photo | anime | illustration | painting | digital_art | screenshot | document | mixed | unknown",
  "final_tags": ["only tags from candidate list"],
  "objects": ["only tags from candidate list"],
  "scene": ["only tags from candidate list"],
  "medium": ["only tags from candidate list"],
  "style": ["only tags from candidate list"],
  "composition": ["only tags from candidate list"],
  "attributes": ["only tags from candidate list"],
  "actions": ["only tags from candidate list"],
  "booru": ["only booru tags from candidate list"],
  "characters": ["only character tags from candidate list"],
  "visible_text": ["visible text snippets, may be copied from semantic analysis"],
  "abstract_tags": ["only tags from candidate list"],
  "rejected_tags": [
    {{"tag": "candidate tag", "reason": "short reason"}}
  ]
}}

Limit rejected_tags to at most {max_rejected} items.

Semantic analysis:
{json.dumps(semantic, ensure_ascii=False)}

Candidate tags:
{json.dumps(candidate_payload, ensure_ascii=False)}
""".strip()


def normalize_candidate_tag(tag: str) -> str:
    cleaned = tag.strip().lower()
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" ,.;:|")
    return TAG_ALIASES.get(cleaned, cleaned)


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(cleaned[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {
        "parse_error": True,
        "raw_text": text.strip(),
    }


def source_authority(source: str, namespace: str) -> float:
    if source == "qwen":
        if namespace in {"medium", "style", "scene", "composition", "abstract", "visible_text"}:
            return 0.95
        return 0.82
    if source == "ram++":
        if namespace in {"object", "objects", "scene"}:
            return 0.90
        return 0.72
    if source == "smilingwolf":
        if namespace in {"booru", "booru_general", "pose", "clothing", "composition"}:
            return 0.86
        if namespace in {"character", "characters"}:
            return 0.72
        if namespace == "rating":
            return 0.65
        return 0.76
    if source == "metadata":
        return 0.88
    return 0.5


def make_candidate(
    raw: str,
    source: str,
    namespace: str,
    confidence: float | None,
) -> CandidateTag | None:
    raw = str(raw).strip()
    tag = normalize_candidate_tag(raw)
    if not tag:
        return None
    authority = source_authority(source, namespace)
    confidence_value = 0.75 if confidence is None else float(confidence)
    score = max(0.0, min(1.0, confidence_value)) * authority
    return CandidateTag(
        raw=raw,
        tag=tag,
        source=source,
        namespace=namespace,
        confidence=confidence,
        authority=authority,
        score=score,
    )


def candidates_from_context(tags: list[str]) -> list[CandidateTag]:
    candidates: list[CandidateTag] = []
    for tag in tags:
        candidate = make_candidate(tag, "metadata", "existing", confidence=0.92)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def candidates_from_ram(ram_result: dict[str, Any]) -> list[CandidateTag]:
    candidates: list[CandidateTag] = []
    for tag in ram_result.get("tags", []):
        candidate = make_candidate(tag, "ram++", "objects", confidence=0.78)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def candidates_from_pipeline_smilingwolf(sw_result: dict[str, Any]) -> list[CandidateTag]:
    candidates: list[CandidateTag] = []
    rating = sw_result.get("rating")
    if isinstance(rating, dict):
        candidate = make_candidate(
            rating.get("tag") or rating.get("raw") or "",
            "smilingwolf",
            "rating",
            rating.get("confidence"),
        )
        if candidate is not None:
            candidates.append(candidate)
    for key, namespace in (
        ("general", "booru"),
        ("characters", "characters"),
        ("copyrights", "copyright"),
        ("meta", "booru"),
    ):
        for item in sw_result.get(key, []):
            if not isinstance(item, dict):
                continue
            candidate = make_candidate(
                item.get("tag") or item.get("raw") or "",
                "smilingwolf",
                namespace,
                item.get("confidence"),
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def candidates_from_qwen_semantic(semantic: dict[str, Any]) -> list[CandidateTag]:
    key_to_namespace = {
        "medium": "medium",
        "scene": "scene",
        "objects": "objects",
        "style": "style",
        "composition": "composition",
        "attributes": "attributes",
        "actions": "actions",
        "abstract_tags": "abstract",
    }
    candidates: list[CandidateTag] = []
    content_type = semantic.get("content_type")
    if isinstance(content_type, str) and content_type.strip() and content_type != "unknown":
        candidate = make_candidate(content_type, "qwen", "medium", confidence=0.82)
        if candidate is not None:
            candidates.append(candidate)
    for key, namespace in key_to_namespace.items():
        value = semantic.get(key)
        if isinstance(value, list):
            for tag in value:
                candidate = make_candidate(str(tag), "qwen", namespace, confidence=0.82)
                if candidate is not None:
                    candidates.append(candidate)
    return candidates


def merge_candidates(candidates: list[CandidateTag], max_candidates: int) -> list[CandidateTag]:
    best: dict[str, CandidateTag] = {}
    for candidate in candidates:
        existing = best.get(candidate.tag)
        if existing is None or candidate.score > existing.score:
            best[candidate.tag] = candidate
    merged = list(best.values())
    merged.sort(key=lambda candidate: candidate.score, reverse=True)
    return merged[:max_candidates]


def build_source_tags(candidates: list[CandidateTag]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for candidate in candidates:
        output.setdefault(candidate.source, []).append(candidate.tag)
    return {
        source: unique_preserve_order(tags)
        for source, tags in sorted(output.items())
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return unique_preserve_order(
        [normalize_candidate_tag(str(item)) for item in value if str(item).strip()]
    )


def _stage_error(stage: str, error: Exception) -> dict[str, str]:
    return {
        "stage": stage,
        "error": repr(error),
        "traceback": traceback.format_exc(),
    }


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


def _pad_to_square(image: Image.Image, fill: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def _patch_transformers_for_ram_plus() -> None:
    try:
        import transformers.modeling_utils as modeling_utils
        import transformers.pytorch_utils as pytorch_utils
    except ImportError:
        return
    for name in (
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    ):
        if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))


def _clear_cuda(torch_module: object) -> None:
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
        try:
            torch_module.cuda.ipc_collect()
        except RuntimeError:
            pass


def _append_tag(tags: list[str], seen: set[str], tag: str) -> None:
    if tag in seen:
        return
    tags.append(tag)
    seen.add(tag)
