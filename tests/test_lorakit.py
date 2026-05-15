import json
from pathlib import Path

import pytest
import torch
from PIL import Image

import lorakit.models as models_module
import lorakit.prepare as prepare_module
import lorakit.tools.tagging as tagging_module
import lorakit.training as training_module
from lorakit.config import CONFIG_NAME, discover_config
from lorakit import candidates as candidates_module
from lorakit import PrepareConfig, Project, TrainingSpec
from lorakit.cli import main
from lorakit.errors import (
    DatasetExists,
    DatasetNotFound,
    InvalidPrepareConfig,
    ModelAmbiguous,
    LorakitError,
)
from lorakit.importers import six2one
from lorakit.manifest import MANIFEST_NAME, read_manifest
from lorakit.paths import Paths
from lorakit.training.backends.types import BackendResult


class FakeWatermarkRemover:
    def process_file(self, input_path: Path, output_path: Path) -> str:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(input_path.read_bytes())
        return "copied unchanged"


@pytest.fixture(autouse=True)
def isolated_lorakit_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@pytest.fixture(autouse=True)
def fake_watermark_remover(monkeypatch):
    monkeypatch.setattr(
        prepare_module,
        "build_watermark_remover",
        lambda **_: FakeWatermarkRemover(),
    )


def test_project_creates_data_layout(tmp_path):
    project = Project(tmp_path / "data")
    shared_models = tmp_path / "home" / ".lorakit" / "models"

    assert project.paths.candidates.exists()
    assert project.paths.staged.exists()
    assert project.paths.prepared.exists()
    assert project.paths.models.exists()
    assert project.paths.models == shared_models
    assert not (tmp_path / "data" / "models").exists()
    assert not (tmp_path / "models").exists()
    assert project.paths.artifacts.exists()


def test_models_install_creates_global_model_paths(tmp_path, monkeypatch):
    paths = Paths(
        tmp_path / "data",
        models_directory=tmp_path / "models",
        huggingface_cache_directory=tmp_path / "models" / "cache",
    )

    result = models_module.install(paths, with_models=False)

    assert result == tmp_path / "models"
    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "models" / "cache").is_dir()


def test_models_install_with_models_installs_watermark_models_in_configured_cache(
    tmp_path,
    monkeypatch,
):
    paths = Paths(
        tmp_path / "data",
        models_directory=tmp_path / "models",
        huggingface_cache_directory=tmp_path / "models" / "cache",
    )
    watermark_calls = []
    file_calls = []
    repo_calls = []

    monkeypatch.setattr(
        models_module,
        "_cache_hf_file",
        lambda *args: file_calls.append(args),
    )
    monkeypatch.setattr(
        models_module,
        "_cache_hf_repo",
        lambda *args: repo_calls.append(args),
    )
    monkeypatch.setattr(
        models_module,
        "ensure_watermark_models",
        lambda **kwargs: watermark_calls.append(kwargs),
    )

    models_module.install(paths, with_models=True)

    assert watermark_calls == [
        {
            "models_dir": tmp_path / "models",
            "cache_dir": tmp_path / "models" / "cache",
        }
    ]
    assert [
        (repo_id, filename)
        for _, repo_id, filename in file_calls
    ] == [
        (models_module.SD15_REPO_ID, models_module.SD15_FILENAME),
        (models_module.SMILINGWOLF_REPO_ID, models_module.SMILINGWOLF_MODEL_FILE),
        (models_module.SMILINGWOLF_REPO_ID, models_module.SMILINGWOLF_TAGS_FILE),
        (models_module.RAM_PLUS_REPO_ID, models_module.RAM_PLUS_FILENAME),
    ]
    assert [repo_id for _, repo_id in repo_calls] == [
        models_module.DEFAULT_PIPELINE_SMILINGWOLF_MODEL,
        models_module.FLORENCE_PROMPTGEN_REPO_ID,
        models_module.DEFAULT_CAPTION_EDITOR_MODEL,
        models_module.DEFAULT_QWEN_VL_MODEL,
    ]
    assert not any((tmp_path / "models" / name).exists() for name in ("sd15", "snapshot"))


def test_init_creates_project_config_and_commands_discover_it(tmp_path, monkeypatch):
    project_dir = tmp_path / "fox-project"

    assert main(["init", str(project_dir)]) == 0

    config_path = project_dir / CONFIG_NAME
    assert config_path.exists()
    assert "name: fox-project" in config_path.read_text(encoding="utf-8")
    discovered = discover_config(project_dir / "nested" / "folder")
    assert discovered is not None
    assert discovered.root == project_dir
    assert discovered.data_dir == project_dir / "data"
    assert discovered.models_dir == Path.home() / ".lorakit" / "models"
    assert discovered.huggingface_cache_dir == Path.home() / ".lorakit" / "models" / "cache"
    config_path.write_text(
        "\n".join(
            [
                "version: 1",
                "name: fox-project",
                "paths:",
                "  data: data",
                f"  models: {tmp_path / 'global-models'}",
                f"  huggingface_cache: {tmp_path / 'global-models' / 'cache'}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    nested = project_dir / "nested" / "folder"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    assert main(["dataset", "create", "ds"]) == 0
    assert (project_dir / "data" / "staged" / "ds").exists()


def test_candidates_list_broken_and_show_metadata(tmp_path):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "good", tags=["fox", "solo"])
    _image(paths.candidates / "image-only.png")
    _metadata(paths.candidates / "json-only.json", tags=["tag"])

    candidates = Project(paths.root).candidates.list()

    assert [(candidate.stem, candidate.is_broken) for candidate in candidates] == [
        ("good", False),
        ("image-only", True),
        ("json-only", True),
    ]
    assert Project(paths.root).candidates.show("good") == {
        "metadata": {"source": "test"},
        "tags": ["fox", "solo"],
    }


def test_candidates_tag_only_missing_tags_by_default_and_merges_with_all(tmp_path):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "tagged", tags=["existing"])
    _image(paths.candidates / "untagged.png")
    project = Project(paths.root)

    default_results = candidates_module.tag(paths, tagger=FakeTagger(["new_tag", "solo"]))

    assert [(result.stem, result.skipped, result.added_tags) for result in default_results] == [
        ("tagged", True, []),
        ("untagged", False, ["new_tag", "solo"]),
    ]
    assert json.loads((paths.candidates / "untagged.json").read_text(encoding="utf-8"))[
        "tags"
    ] == ["new_tag", "solo"]

    all_results = candidates_module.tag(
        paths,
        all_images=True,
        tagger=FakeTagger(["new_tag", "solo"]),
    )

    assert [(result.stem, result.skipped) for result in all_results] == [
        ("tagged", False),
        ("untagged", False),
    ]
    assert json.loads((paths.candidates / "tagged.json").read_text(encoding="utf-8"))[
        "tags"
    ] == ["existing", "new_tag", "solo"]


def test_candidates_caption_presets_use_composite_tagger(tmp_path, monkeypatch):
    paths = Paths(tmp_path / "data")
    _image(paths.candidates / "0001.png")

    def fake_build_tagger(*, presets):
        assert presets == ["natural_language", "image_tags"]
        return FakeTagger(["smilingwolf_tag", "natural language tag"])

    monkeypatch.setattr(candidates_module, "build_tagger", fake_build_tagger)

    results = Project(paths.root).candidates.caption(
        presets=["natural_language", "image_tags"]
    )

    assert results[0].added_tags == ["smilingwolf_tag", "natural language tag"]
    assert json.loads((paths.candidates / "0001.json").read_text(encoding="utf-8"))[
        "tags"
    ] == ["smilingwolf_tag", "natural language tag"]


def test_candidates_caption_passes_existing_tags_as_model_context(tmp_path):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "0001", tags=["anthro girl", "beige room"])
    tagger = FakeTagger(["an anthropomorphic girl in a beige room"])

    candidates_module.tag(paths, all_images=True, tagger=tagger, quiet=True)

    assert tagger.contexts == [["anthro girl", "beige room"]]
    assert json.loads((paths.candidates / "0001.json").read_text(encoding="utf-8"))[
        "tags"
    ] == [
        "anthro girl",
        "beige room",
        "an anthropomorphic girl in a beige room",
    ]


def test_candidates_caption_writes_top_level_caption_and_captioning_metadata(tmp_path):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "0001", tags=["anthro", "fox"])
    tagger = FakeStructuredTagger(
        tagging_module.TaggingResult(
            tags=[],
            caption="An anthropomorphic fox stands alone.",
            draft_caption="A lemur-like character stands alone.",
            caption_backend="Florence-2-large-PromptGen",
            editor_backend="Dolphin3.0-Llama3.1-8B",
        )
    )

    candidates_module.tag(paths, all_images=True, tagger=tagger, quiet=True)

    metadata = json.loads((paths.candidates / "0001.json").read_text(encoding="utf-8"))
    assert metadata["tags"] == ["anthro", "fox"]
    assert metadata["caption"] == "An anthropomorphic fox stands alone."
    assert metadata["metadata"]["captioning"] == {
        "draft_caption": "A lemur-like character stands alone.",
        "caption": "An anthropomorphic fox stands alone.",
        "caption_backend": "Florence-2-large-PromptGen",
        "editor_backend": "Dolphin3.0-Llama3.1-8B",
    }


def test_candidates_caption_writes_pipeline_sidecar_and_final_tags(tmp_path):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "0001", tags=["anthro"])
    sidecar = {
        "semantic": {"caption": "A rough description."},
        "final": {
            "caption": "An anthropomorphic fox stands alone.",
            "final_tags": ["anthro", "fox", "solo"],
        },
        "source_tags": {"qwen": ["fox", "solo"]},
        "errors": [],
    }
    tagger = FakeStructuredTagger(
        tagging_module.TaggingResult(
            tags=["fox", "solo"],
            caption="An anthropomorphic fox stands alone.",
            draft_caption="A rough description.",
            caption_backend="Qwen2.5-VL-7B-Instruct",
            editor_backend="Qwen2.5-VL-7B-Instruct",
            sidecar=sidecar,
        )
    )

    candidates_module.tag(paths, all_images=True, tagger=tagger, quiet=True)

    metadata = json.loads((paths.candidates / "0001.json").read_text(encoding="utf-8"))
    assert metadata["tags"] == ["anthro", "fox", "solo"]
    assert metadata["caption"] == "An anthropomorphic fox stands alone."
    assert metadata["metadata"]["tagging"] == sidecar


def test_candidates_caption_writes_pipeline_results_iteratively(tmp_path):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "0001", tags=["existing"])
    _candidate(paths, "0002", tags=["existing"])
    tagger = FakeIterativeStructuredTagger(paths.candidates)

    results = candidates_module.tag(paths, all_images=True, tagger=tagger, quiet=True)

    assert [result.stem for result in results] == ["0001", "0002"]
    assert tagger.first_saved_before_second is True
    first = json.loads((paths.candidates / "0001.json").read_text(encoding="utf-8"))
    second = json.loads((paths.candidates / "0002.json").read_text(encoding="utf-8"))
    assert first["tags"] == ["existing", "tag-0"]
    assert first["caption"] == "Caption 0."
    assert first["metadata"]["tagging"] == {"index": 0}
    assert second["tags"] == ["existing", "tag-1"]
    assert second["caption"] == "Caption 1."
    assert second["metadata"]["tagging"] == {"index": 1}


def test_pipeline_final_tags_fall_back_to_model_outputs_when_qwen_returns_empty():
    merged = [
        tagging_module.CandidateTag(
            raw="animal",
            tag="animal",
            source="ram++",
            namespace="objects",
            confidence=0.78,
            authority=0.9,
            score=0.7,
        ),
        tagging_module.CandidateTag(
            raw="potty",
            tag="potty",
            source="ram++",
            namespace="objects",
            confidence=0.78,
            authority=0.9,
            score=0.7,
        ),
    ]

    assert tagging_module._final_tags_or_model_output_fallback(
        {"final_tags": []},
        merged,
    ) == ["animal", "potty"]
    assert tagging_module._final_tags_or_model_output_fallback(
        {"final_tags": ["qwen tag"]},
        merged,
    ) == ["qwen tag"]


def test_pipeline_final_tags_always_include_ram_tags():
    assert tagging_module._with_required_tags(
        ["qwen tag", "potty"],
        ["potty", "toddler", "floor"],
    ) == ["qwen tag", "potty", "toddler", "floor"]
    assert tagging_module._ram_tags_from_candidate_record(
        {"ram++": {"tags": ["potty", "toddler"]}}
    ) == ["potty", "toddler"]


def test_pipeline_content_type_becomes_required_tag():
    assert tagging_module._content_type_tags({"content_type": "digital_art"}) == [
        "digital art"
    ]
    assert tagging_module._content_type_tags({"content_type": "photo"}) == ["photo"]
    assert tagging_module._content_type_tags({"content_type": "unknown"}) == []


def test_pipeline_semantic_tags_become_required_tags():
    semantic = {
        "content_type": "digital_art",
        "medium": ["illustration"],
        "scene": ["forest"],
        "objects": ["tree"],
        "style": ["soft_shading"],
        "composition": ["close-up"],
        "attributes": ["blue_eyes"],
        "actions": ["standing"],
        "abstract_tags": ["calm"],
        "visible_text": ["do not tag this"],
        "uncertain": ["also skipped"],
    }

    assert tagging_module._semantic_tags_from_result(semantic) == [
        "digital art",
        "illustration",
        "forest",
        "tree",
        "soft shading",
        "close-up",
        "blue eyes",
        "standing",
        "calm",
    ]


def test_pipeline_skips_smilingwolf_for_photo_semantics(monkeypatch):
    tagger = tagging_module.PipelineTagger.__new__(tagging_module.PipelineTagger)
    tagger._device = "cpu"
    tagger._torch = FakeTorchModule()
    request = tagging_module.TagRequest(Path("photo.jpg"), [])
    candidate_record = {"ram++": {"tags": ["potty"]}, "smilingwolf": None, "errors": []}

    def fail_load(*args, **kwargs):
        raise AssertionError("SmilingWolf should not load for photo content")

    monkeypatch.setattr(tagging_module, "_load_pipeline_source", fail_load)

    tagger._smilingwolf_stage(
        requests=[request],
        candidate_records=[candidate_record],
        semantic_records=[{"content_type": "photo"}],
        quiet=True,
    )

    assert candidate_record["smilingwolf"] is None


def test_pipeline_uses_smilingwolf_for_digital_art_semantics(monkeypatch):
    tagger = tagging_module.PipelineTagger.__new__(tagging_module.PipelineTagger)
    tagger._device = "cpu"
    tagger._torch = FakeTorchModule()
    request = tagging_module.TagRequest(Path("art.png"), [])
    candidate_record = {"ram++": {"tags": ["fox"]}, "smilingwolf": None, "errors": []}

    class FakeSmilingWolf:
        def tag(self, image_path):
            return {"general": [{"tag": "anthro", "confidence": 0.9}]}

        def unload(self):
            self.unloaded = True

    def fake_load(name, source_class, *, device):
        assert name == "smilingwolf"
        return FakeSmilingWolf(), None

    monkeypatch.setattr(tagging_module, "_load_pipeline_source", fake_load)

    tagger._smilingwolf_stage(
        requests=[request],
        candidate_records=[candidate_record],
        semantic_records=[{"content_type": "digital_art"}],
        quiet=True,
    )

    assert candidate_record["smilingwolf"] == {
        "general": [{"tag": "anthro", "confidence": 0.9}]
    }


def test_pipeline_sidecar_omits_raw_qwen_responses():
    assert tagging_module._without_raw_response(
        {"caption": "A caption.", "_raw_response": "raw"}
    ) == {"caption": "A caption."}


def test_florence_promptgen_uses_hf_compatible_image_text_model(monkeypatch):
    captured: dict[str, object] = {}

    class FakeTorch:
        float32 = object()

        class cuda:
            @staticmethod
            def is_available():
                return False

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, repo_id, **kwargs):
            captured["processor"] = (repo_id, kwargs)
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, repo_id, **kwargs):
            captured["model"] = (repo_id, kwargs)
            return cls()

        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            captured["eval"] = True
            return self

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            return FakeTorch
        if name == "transformers":
            class FakeTransformers:
                AutoModelForImageTextToText = FakeModel
                AutoProcessor = FakeProcessor

            return FakeTransformers
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    tagger = tagging_module.FlorencePromptGenTagger()
    assert "model" not in captured

    tagger._model_for_generation()

    _, model_kwargs = captured["model"]
    assert captured["model"][0] == "Disty0/Florence-2-large-PromptGen-v2.0"
    assert captured["processor"][0] == "Disty0/Florence-2-large-PromptGen-v2.0"
    assert model_kwargs == {"cache_dir": models_module.HF_CACHE_DIR}
    assert captured["device"] == "cpu"
    assert captured["eval"] is True


def test_caption_editor_limits_tag_context():
    tags = [f"very specific tag {index}" for index in range(200)]

    selected = tagging_module._caption_editor_tags(tags)
    text = ", ".join(selected)

    assert len(text) <= 4000
    assert "very specific tag 0" in text
    assert "very specific tag 199" not in text


def test_caption_editor_prompt_requires_natural_language():
    prompt = tagging_module.CAPTION_EDITOR_SYSTEM_PROMPT

    assert "Output natural language only." in prompt
    assert "Do not output booru tags" in prompt
    assert "Every trusted tag that describes visible content should be represented" in prompt
    assert "Adult and explicit trusted tags are not optional" in prompt


def test_clean_caption_output_removes_labels_and_explanations():
    assert (
        tagging_module.clean_caption_output(
            'Corrected caption: "anthro fox, solo. A fox in a forest."\n\nI changed it.'
        )
        == "anthro fox, solo. A fox in a forest."
    )


def test_natural_language_preset_uses_pipeline_tagger(monkeypatch):
    created = {}

    class FakePipelineTagger:
        def __init__(self):
            created["pipeline"] = True

    monkeypatch.setattr(tagging_module, "PipelineTagger", FakePipelineTagger)

    tagger = tagging_module.build_tagger(presets=["natural_language", "image_tags"])

    assert isinstance(tagger, FakePipelineTagger)
    assert created == {"pipeline": True}


def test_qwen_chat_image_uses_plain_paths_for_plus_filenames(tmp_path):
    captured: dict[str, object] = {}
    image_path = tmp_path / "best+toddler+potties.webp"
    _image(image_path)
    judge = tagging_module.QwenVLTagJudge.__new__(tagging_module.QwenVLTagJudge)
    judge._device = "cpu"
    judge._torch = FakeTorchModule()
    judge._model = FakeGenerateModel()
    judge._processor = FakeQwenProcessor()

    def fake_process_vision_info(messages):
        captured["messages"] = messages
        return ["image"], None

    judge._process_vision_info = fake_process_vision_info

    judge._chat_image(image_path, "describe", max_new_tokens=4)

    messages = captured["messages"]
    image_value = messages[0]["content"][0]["image"]
    assert image_value == str(image_path.resolve())
    assert "%2B" not in image_value
    assert not image_value.startswith("file://")


def test_candidates_tag_limit_bounds_tagged_images(tmp_path):
    paths = Paths(tmp_path / "data")
    _image(paths.candidates / "0001.png")
    _image(paths.candidates / "0002.png")

    results = candidates_module.tag(paths, limit=1, tagger=FakeTagger(["tagged"]))

    assert [(result.stem, result.skipped) for result in results] == [("0001", False)]
    assert (paths.candidates / "0001.json").exists()
    assert not (paths.candidates / "0002.json").exists()


def test_dataset_lifecycle_stage_status_prepare_flags_and_delete(tmp_path):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "0001")
    project = Project(paths.root)

    project.datasets.create("fox-solo")
    staged = project.datasets.stage("fox-solo", "0001")
    assert staged.exists()
    assert project.datasets.list()[0].staged_images == 1

    status = project.datasets.status("fox-solo")
    assert status.staged_images == 1
    assert status.missing_metadata == 0
    assert status.prepared is False

    project.datasets.prepare("fox-solo", PrepareConfig(width=32, height=32))
    assert project.datasets.status("fox-solo").prepared is True

    removed = project.datasets.unstage("fox-solo", "0001")
    assert removed.name == "0001.png"
    project.datasets.rename("fox-solo", "renamed")
    with pytest.raises(DatasetExists):
        project.datasets.create("renamed")
    project.datasets.delete("renamed")
    with pytest.raises(DatasetNotFound):
        project.datasets.status("renamed")


def test_stage_can_symlink_and_replaces_existing_file(tmp_path):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "0001")
    project = Project(paths.root)
    project.datasets.create("ds")

    first = project.datasets.stage("ds", "0001")
    second = project.datasets.stage("ds", paths.candidates / "0001.png", symlink=True)

    assert first == second
    assert second.is_symlink()


def test_stage_all_adds_every_candidate_and_rejects_broken_entries(tmp_path):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "0001")
    _candidate(paths, "0002")
    project = Project(paths.root)
    project.datasets.create("ds")

    staged = project.datasets.stage_all("ds")

    assert [path.name for path in staged] == ["0001.png", "0002.png"]
    assert project.datasets.status("ds").staged_images == 2

    _image(paths.candidates / "broken.png")
    with pytest.raises(LorakitError, match="broken entries: broken"):
        project.datasets.stage_all("ds")


def test_clean_reports_and_deletes_candidate_and_staged_orphans(tmp_path):
    paths = Paths(tmp_path / "data")
    project = Project(paths.root)
    _image(paths.candidates / "no-json.png")
    _metadata(paths.candidates / "no-image.json")
    project.datasets.create("ds")
    _image(paths.staged_for("ds") / "staged-no-json.png")
    _metadata(paths.staged_for("ds") / "override-no-image.json")

    report = project.clean()
    assert len(report.orphans) == 4
    assert report.deleted == []

    applied = project.clean(apply=True)
    assert len(applied.deleted) == 4
    assert all(not path.exists() for path in applied.deleted)


def test_prepare_writes_manifest_and_supports_fit_center_crop_pad_and_copy(tmp_path):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "wide", size=(100, 50), tags=["wide"])
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "wide")

    prepared = project.datasets.prepare(
        "ds",
        PrepareConfig(width=50, height=50, trigger="lorakit"),
    )
    assert _image_size(prepared / "images" / "wide.png") == (50, 25)
    assert read_manifest(prepared / MANIFEST_NAME) == [
        {
            "caption": "lorakit, wide. lorakit, wide",
            "image": "images/wide.png",
            "tags": ["wide"],
        }
    ]

    prepared = project.datasets.prepare(
        "ds",
        PrepareConfig(mode="center-crop", width=40, height=40),
    )
    assert _image_size(prepared / "images" / "wide.png") == (40, 40)

    prepared = project.datasets.prepare("ds", PrepareConfig(mode="pad", width=40, height=40))
    assert _image_size(prepared / "images" / "wide.png") == (40, 40)

    prepared = project.datasets.prepare("ds", PrepareConfig(mode="copy", width=None, height=None))
    assert _image_size(prepared / "images" / "wide.png") == (100, 50)


def test_prepare_uses_watermark_remover_by_default_and_can_disable_it(tmp_path, monkeypatch):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "item")
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "item")
    calls = []

    def fake_builder(**kwargs):
        calls.append(kwargs)
        return FakeWatermarkRemover()

    monkeypatch.setattr(prepare_module, "build_watermark_remover", fake_builder)

    project.datasets.prepare("ds", PrepareConfig(width=32, height=32))
    assert len(calls) == 1
    assert calls[0]["models_dir"] == paths.models
    assert calls[0]["cache_dir"] == paths.huggingface_cache
    assert calls[0]["allow_download"] is False

    project.datasets.prepare(
        "ds",
        PrepareConfig(width=32, height=32, remove_watermarks=False),
    )
    assert len(calls) == 1


def test_prepare_uses_staged_metadata_override(tmp_path):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "item", tags=["candidate"])
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "item")
    _metadata(paths.staged_for("ds") / "item.json", tags=["override"])

    prepared = project.datasets.prepare("ds", PrepareConfig(width=32, height=32))

    assert read_manifest(prepared / MANIFEST_NAME)[0]["caption"] == "override. override"


def test_prepare_expands_underscore_tags_from_candidate_and_override_metadata(tmp_path):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "candidate", tags=["hi_res", "solo"])
    _candidate(paths, "override", tags=["ignored"])
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "candidate")
    project.datasets.stage("ds", "override")
    _metadata(paths.staged_for("ds") / "override.json", tags=["blue_eyes", "blue eyes"])

    prepared = project.datasets.prepare("ds", PrepareConfig(width=32, height=32))
    rows = read_manifest(prepared / MANIFEST_NAME)

    assert rows[0]["tags"] == ["hi_res", "hi res", "solo"]
    assert rows[0]["caption"] == "hi_res, solo. hi res, solo"
    assert rows[1]["tags"] == ["blue_eyes", "blue eyes"]
    assert rows[1]["caption"] == "blue_eyes, blue eyes. blue eyes, blue eyes"


def test_prepare_prompt_type_modes_use_tags_natural_caption_and_all(tmp_path):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "item", tags=["anthro", "blue_eyes", "looking_at_viewer"])
    metadata_path = paths.candidates / "item.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["caption"] = "An anthropomorphic character with blue eyes looks at the viewer."
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "item")

    prepared = project.datasets.prepare(
        "ds",
        PrepareConfig(width=32, height=32, trigger="iztli", prompt_type="tags"),
    )
    row = read_manifest(prepared / MANIFEST_NAME)[0]
    assert row["caption"] == (
        "iztli, anthro, blue_eyes, looking_at_viewer"
    )
    assert row["tags"] == ["anthro", "blue_eyes", "looking_at_viewer"]

    prepared = project.datasets.prepare(
        "ds",
        PrepareConfig(width=32, height=32, trigger="iztli", prompt_type="natural"),
    )
    row = read_manifest(prepared / MANIFEST_NAME)[0]
    assert row["caption"] == (
        "iztli, anthro, blue eyes, looking at viewer"
    )
    assert row["tags"] == ["anthro", "blue eyes", "looking at viewer"]

    prepared = project.datasets.prepare(
        "ds",
        PrepareConfig(width=32, height=32, trigger="iztli", prompt_type="caption"),
    )
    row = read_manifest(prepared / MANIFEST_NAME)[0]
    assert row["caption"] == (
        "iztli. An anthropomorphic character with blue eyes looks at the viewer."
    )
    assert row["tags"] == ["anthro", "blue eyes", "looking at viewer"]

    prepared = project.datasets.prepare(
        "ds",
        PrepareConfig(width=32, height=32, trigger="iztli", prompt_type="all"),
    )
    row = read_manifest(prepared / MANIFEST_NAME)[0]
    assert row["caption"] == (
        "iztli, anthro, blue_eyes, looking_at_viewer. "
        "iztli, anthro, blue eyes, looking at viewer. "
        "iztli. An anthropomorphic character with blue eyes looks at the viewer."
    )
    assert row["tags"] == ["anthro", "blue_eyes", "blue eyes", "looking_at_viewer", "looking at viewer"]


def test_prepare_caption_prompt_type_falls_back_to_natural_tags(tmp_path):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "item", tags=["blue_eyes"])
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "item")

    prepared = project.datasets.prepare(
        "ds",
        PrepareConfig(width=32, height=32, prompt_type="caption"),
    )

    assert read_manifest(prepared / MANIFEST_NAME)[0]["caption"] == "blue eyes"


def test_prepare_flattens_six2one_tag_categories(tmp_path):
    paths = Paths(tmp_path / "data")
    paths.ensure()
    _image(paths.candidates / "0001.png")
    (paths.candidates / "0001.json").write_text(
        json.dumps(
            {
                "tags": {
                    "artist": ["artist_name"],
                    "general": ["anthro", "solo"],
                    "species": ["fox"],
                    "meta": ["hi_res"],
                }
            }
        ),
        encoding="utf-8",
    )
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "0001")

    prepared = project.datasets.prepare("ds", PrepareConfig(width=32, height=32))

    assert read_manifest(prepared / MANIFEST_NAME)[0]["tags"] == [
        "anthro",
        "solo",
        "fox",
        "artist_name",
        "artist name",
        "hi_res",
        "hi res",
    ]


def test_prepare_validates_missing_metadata_and_transparent_jpg(tmp_path):
    paths = Paths(tmp_path / "data")
    project = Project(paths.root)
    project.datasets.create("ds")
    _image(paths.staged_for("ds") / "lonely.png")
    with pytest.raises(LorakitError):
        project.datasets.prepare("ds", PrepareConfig(width=32, height=32))

    _metadata(paths.candidates / "alpha.json")
    _image(paths.candidates / "alpha.png", mode="RGBA", color=(0, 0, 0, 0))
    project.datasets.stage("ds", "alpha")
    with pytest.raises(InvalidPrepareConfig):
        project.datasets.prepare("ds", PrepareConfig(image_format="jpg"))


def test_prepare_rejects_copy_with_dimensions_and_non_original_format(tmp_path):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "0001")
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "0001")

    with pytest.raises(InvalidPrepareConfig):
        project.datasets.prepare("ds", PrepareConfig(mode="copy", width=32, height=None))
    with pytest.raises(InvalidPrepareConfig):
        project.datasets.prepare(
            "ds",
            PrepareConfig(mode="copy", width=None, height=None, image_format="png"),
        )


def test_prepare_rejects_converted_filename_collisions(tmp_path):
    paths = Paths(tmp_path / "data")
    project = Project(paths.root)
    project.datasets.create("ds")
    _image(paths.staged_for("ds") / "same.png")
    _image(paths.staged_for("ds") / "same.jpg")
    _metadata(paths.staged_for("ds") / "same.json")

    with pytest.raises(LorakitError, match="filename collision"):
        project.datasets.prepare("ds", PrepareConfig(image_format="png"))


def test_prepare_dimension_rules_for_single_dimension_modes(tmp_path):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "tall", size=(20, 100))
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "tall")

    prepared = project.datasets.prepare("ds", PrepareConfig(width=40, height=None))
    assert _image_size(prepared / "images" / "tall.png") == (40, 200)

    prepared = project.datasets.prepare(
        "ds",
        PrepareConfig(mode="pad", width=None, height=40),
    )
    assert _image_size(prepared / "images" / "tall.png") == (40, 40)


def test_models_list_resolve_remove_search_and_fetch(tmp_path, monkeypatch):
    paths = Paths(tmp_path / "data")
    project = Project(paths.root)
    (paths.models / "sd15.safetensors").write_bytes(b"model")
    (paths.models / "pony.ckpt").write_bytes(b"model")
    (paths.models / "diffusers").mkdir()
    (paths.models / "diffusers" / "model_index.json").write_text("{}", encoding="utf-8")
    (paths.models / "not-a-model").mkdir()

    listed = project.models.list()
    assert [model.name for model in listed] == ["diffusers", "pony", "sd15"]
    assert project.models.resolve("sd15").path == paths.models / "sd15.safetensors"
    assert project.models.resolve(str(paths.models / "pony.ckpt")).source == "path"
    assert project.models.resolve("owner/repo").repo_id == "owner/repo"
    assert project.models.resolve("missing").repo_id == "missing"
    assert project.models.resolve("not-a-model").path == paths.models / "not-a-model"

    (paths.models / "ambiguous.ckpt").write_bytes(b"model")
    (paths.models / "ambiguous.pt").write_bytes(b"model")
    with pytest.raises(ModelAmbiguous):
        project.models.resolve("ambiguous")

    class FakeModel:
        modelId = "owner/model"
        author = "owner"
        downloads = 7
        likes = 3

    class FakeApi:
        def list_models(self, *, search, pipeline_tag, limit):
            assert (search, pipeline_tag, limit) == ("fox", "text-to-image", 1)
            return [FakeModel()]

        def list_repo_files(self, *, repo_id):
            assert repo_id == "owner/new-model"
            return ["model.safetensors", "README.md"]

    def fake_hf_hub_download(*, repo_id, filename, cache_dir):
        assert repo_id == "owner/new-model"
        assert filename == "model.safetensors"
        cache_dir.mkdir(parents=True, exist_ok=True)
        source = cache_dir / filename
        source.write_bytes(b"model")
        return source

    monkeypatch.setattr(models_module, "HfApi", FakeApi)
    monkeypatch.setattr(models_module, "hf_hub_download", fake_hf_hub_download)

    assert project.models.search("fox", limit=1)[0]["model_id"] == "owner/model"
    assert project.models.fetch("owner/new-model").name == "model.safetensors"
    assert project.models.remove("pony").name == "pony.ckpt"


def test_models_resolve_sd15_alias_when_no_local_match(tmp_path):
    project = Project(tmp_path / "data")

    assert project.models.resolve("sd15").repo_id == models_module.SD15_REPO_ID


def test_models_fetch_rejects_multiple_safetensors_when_not_interactive(tmp_path, monkeypatch):
    paths = Paths(tmp_path / "data")
    project = Project(paths.root)

    class FakeApi:
        def list_repo_files(self, *, repo_id):
            assert repo_id == "owner/multi"
            return ["a.safetensors", "b.safetensors"]

    monkeypatch.setattr(models_module, "HfApi", FakeApi)

    with pytest.raises(ModelAmbiguous, match="Multiple .safetensors files"):
        project.models.fetch("owner/multi")


def test_clip_text_model_compatibility_patch_removed():
    from transformers import CLIPTextModel

    assert not hasattr(CLIPTextModel, "text_model")


def test_move_module_to_device_handles_meta_tensors(tmp_path):
    from lorakit.training.backends import diffusers as diffusers_backend

    module = torch.nn.Module()
    module.param = torch.nn.Parameter(torch.empty(2, 2, device="meta"))
    module.register_buffer("buf", torch.empty(1, device="meta"))

    moved = diffusers_backend._move_module_to_device(module, torch.device("cpu"), dtype=torch.float32)

    assert moved.param.device == torch.device("cpu")
    assert moved.buf.device == torch.device("cpu")
    assert moved.param.dtype == torch.float32


def test_diffusers_load_rows_rejects_mixed_image_shapes(tmp_path):
    from lorakit.training.backends import diffusers as diffusers_backend

    prepared_dir = tmp_path / "prepared"
    _image(prepared_dir / "images" / "a.png", size=(32, 32))
    _image(prepared_dir / "images" / "b.png", size=(64, 32))
    _write_prepared_manifest(
        prepared_dir,
        [
            {"image": "images/a.png", "caption": "a", "tags": ["tag"]},
            {"image": "images/b.png", "caption": "b", "tags": ["tag"]},
        ],
    )

    with pytest.raises(LorakitError, match="must share one tensor shape"):
        diffusers_backend._load_rows(prepared_dir)


def test_diffusers_disk_cache_dataset_validates_and_loads_tensors(tmp_path):
    from lorakit.training.backends import diffusers as diffusers_backend

    latent_path = tmp_path / "latents" / "00000000.pt"
    hidden_path = tmp_path / "text" / "00000000.pt"
    latent_path.parent.mkdir(parents=True)
    hidden_path.parent.mkdir(parents=True)
    torch.save(torch.zeros(4, 8, 8), latent_path)
    torch.save(torch.zeros(77, 768), hidden_path)
    record = diffusers_backend._CacheRecord(
        image="images/a.png",
        caption_sha256="caption",
        image_sha256="image",
        latent_path=latent_path,
        encoder_hidden_state_path=hidden_path,
        latent_shape=(4, 8, 8),
        encoder_hidden_state_shape=(77, 768),
    )

    dataset = diffusers_backend._DiskCachedLatentDataset(records=[record])

    item = dataset[0]
    assert tuple(item["latents"].shape) == (4, 8, 8)
    assert tuple(item["encoder_hidden_states"].shape) == (77, 768)


def test_models_remove_rejects_non_local_repo_id(tmp_path):
    project = Project(tmp_path / "data")

    with pytest.raises(LorakitError, match="Local model not found"):
        project.models.remove("owner/repo")


def test_models_remove_deletes_fetched_non_diffusers_directory(tmp_path):
    paths = Paths(tmp_path / "data")
    project = Project(paths.root)
    fetched_dir = paths.models / "tiny-gpt2"
    fetched_dir.mkdir()
    (fetched_dir / "config.json").write_text("{}", encoding="utf-8")

    removed = project.models.remove("tiny-gpt2")

    assert removed == fetched_dir
    assert not fetched_dir.exists()


def test_training_dry_run_does_not_write(tmp_path):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "0001")
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "0001")
    (paths.models / "sd15.safetensors").write_bytes(b"model")

    dry = project.train(TrainingSpec(dataset="ds", model="sd15", dry_run=True))
    assert dry.dry_run is True
    assert dry.run_name == "run-001"
    assert dry.artifact_dir.exists() is False

    assert paths.prepared_for("ds").exists() is False


def test_training_invokes_backend_and_archives_artifacts(tmp_path, monkeypatch):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "0001")
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "0001")
    (paths.models / "sd15.safetensors").write_bytes(b"model")
    seen = {}

    class FakeBackend:
        def train(self, spec):
            seen["spec"] = spec
            spec.output_dir.mkdir(parents=True)
            model_path = spec.output_dir / "pytorch_lora_weights.safetensors"
            model_path.write_bytes(b"lora")
            return BackendResult(model_path=model_path)

    monkeypatch.setattr(training_module, "get_backend", lambda name: FakeBackend())

    result = project.train(
        TrainingSpec(dataset="ds", model="sd15", steps=1, mixed_precision="no")
    )

    assert result.run_name == "run-001"
    assert seen["spec"].model == str(paths.models / "sd15.safetensors")
    assert (result.artifact_dir / "dataset" / MANIFEST_NAME).exists()
    assert (result.artifact_dir / "model.safetensors").read_bytes() == b"lora"
    assert not (result.artifact_dir / "working").exists()


def test_training_no_prepare_requires_valid_prepared_dataset(tmp_path):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "0001")
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "0001")
    (paths.models / "sd15.safetensors").write_bytes(b"model")

    with pytest.raises(LorakitError, match="Prepared dataset is missing or invalid"):
        project.train(TrainingSpec(dataset="ds", model="sd15", no_prepare=True))


def test_dataset_delete_preserves_artifacts(tmp_path):
    paths = Paths(tmp_path / "data")
    project = Project(paths.root)
    project.datasets.create("ds")
    artifact_dir = paths.artifacts_for("ds") / "run-001"
    artifact_dir.mkdir(parents=True)

    project.datasets.delete("ds")

    assert artifact_dir.exists()
    assert not paths.staged_for("ds").exists()


def test_importer_imports_pairs_and_skips_existing(tmp_path, monkeypatch):
    paths = Paths(tmp_path / "data")
    source = tmp_path / "download"
    _image(source / "0001.png")
    _six2one_metadata(source / "0001.json", post_id=1)
    _image(source / "broken.png")

    first = six2one.import_pairs(paths, source)
    assert sorted(path.name for path in first.imported) == ["0001.json", "0001.png"]
    assert json.loads((paths.candidates / "0001.json").read_text(encoding="utf-8")) == {
        "metadata": {
            "post_id": 1,
            "source": "e621",
        },
        "tags": ["anthro", "solo", "hi_res"],
    }

    second = six2one.import_pairs(paths, source)
    assert sorted(path.name for path in second.skipped) == ["0001.json", "0001.png"]

def test_importer_overwrite_replaces_existing_files(tmp_path):
    paths = Paths(tmp_path / "data")
    original = tmp_path / "original"
    updated = tmp_path / "updated"
    _image(original / "0001.png", color=(255, 0, 0))
    _six2one_metadata(original / "0001.json", post_id=1, general_tags=["old"])
    _image(updated / "0001.png", color=(0, 255, 0))
    _six2one_metadata(updated / "0001.json", post_id=1, general_tags=["new"])

    six2one.import_pairs(paths, original)
    result = six2one.import_pairs(paths, updated, overwrite=True)

    assert sorted(path.name for path in result.imported) == ["0001.json", "0001.png"]
    assert json.loads((paths.candidates / "0001.json").read_text(encoding="utf-8"))[
        "tags"
    ] == ["new", "hi_res"]

def test_importer_run_shells_out_and_copies_results(tmp_path, monkeypatch):
    paths = Paths(tmp_path / "data")
    monkeypatch.setattr(six2one.Path, "home", lambda: tmp_path)

    def fake_run(command, *, check):
        out_dir = Path(command[-1])
        _six2one_output_post(out_dir, 1)

    monkeypatch.setattr(six2one.subprocess, "run", fake_run)

    result = six2one.run(paths, ["fox", "--safe"], overwrite=False)

    assert sorted(path.name for path in result.imported) == ["000000000001.json", "000000000001.png"]


def test_importer_run_uses_shared_621_cache(tmp_path, monkeypatch):
    paths = Paths(tmp_path / "data")
    monkeypatch.setattr(six2one.Path, "home", lambda: tmp_path)

    captured: dict[str, list[str] | Path] = {}

    def fake_run(command, *, check):
        captured["command"] = command
        out_index = command.index("--out") + 1
        out_dir = Path(command[out_index])
        _six2one_output_post(out_dir, 1)

    monkeypatch.setattr(six2one.subprocess, "run", fake_run)

    result = six2one.run(paths, ["fox"], overwrite=False)

    assert captured["command"][:3] == [
        six2one.sys.executable,
        "-c",
        "from six2one.cli import sync_main; sync_main()",
    ]
    assert "--merge" not in captured["command"]
    assert "--out" in captured["command"]
    assert captured["command"][captured["command"].index("--out") + 1] == str(
        tmp_path / ".lorakit" / "cache" / "621"
    )
    assert sorted(path.name for path in result.imported) == ["000000000001.json", "000000000001.png"]


def test_importer_run_imports_only_new_persistent_cache_pairs(tmp_path, monkeypatch):
    paths = Paths(tmp_path / "data")
    monkeypatch.setattr(six2one.Path, "home", lambda: tmp_path)

    cache_dir = tmp_path / ".lorakit" / "cache" / "621"
    _six2one_output_post(cache_dir, 1)

    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(six2one, "_load_six2one_manifest", lambda source_dir: {})
    monkeypatch.setattr(
        six2one,
        "_post_ids_for_query",
        lambda source_dir, args, manifest: {2},
    )

    def fake_run(command, *, check):
        out_dir = Path(command[command.index("--out") + 1])
        _six2one_output_post(out_dir, 2)

    monkeypatch.setattr(six2one.subprocess, "run", fake_run)

    result = six2one.run(paths, ["fox"], overwrite=False)

    assert sorted(path.name for path in result.imported) == ["000000000002.json", "000000000002.png"]
    assert not (paths.candidates / "000000000001.png").exists()
    assert not (paths.candidates / "000000000001.json").exists()


def test_importer_run_preserves_site_in_candidate_metadata(tmp_path, monkeypatch):
    paths = Paths(tmp_path / "data")
    monkeypatch.setattr(six2one.Path, "home", lambda: tmp_path)

    def fake_run(command, *, check):
        out_dir = Path(command[-1])
        _six2one_output_post(out_dir, 1)

    monkeypatch.setattr(six2one.subprocess, "run", fake_run)

    six2one.run(paths, ["fox", "--site", "e926"], overwrite=False)

    metadata = json.loads((paths.candidates / "000000000001.json").read_text(encoding="utf-8"))
    assert metadata["metadata"]["source"] == "e926"


def test_importer_run_converts_command_failure_to_domain_error(tmp_path, monkeypatch):
    paths = Paths(tmp_path / "data")

    def fake_run(command, *, check):
        raise six2one.subprocess.CalledProcessError(returncode=2, cmd=command)

    monkeypatch.setattr(six2one.subprocess, "run", fake_run)

    with pytest.raises(LorakitError, match="six2one import failed"):
        six2one.run(paths, ["fox"], overwrite=False)


def test_cli_create_list_json_and_error_paths(tmp_path, capsys):
    data_dir = tmp_path / "data"
    assert main(["--data-dir", str(data_dir), "dataset", "create", "ds"]) == 0
    create_output = capsys.readouterr().out
    assert "created dataset ds" in create_output
    assert f"path: {data_dir / 'staged' / 'ds'}" in create_output
    assert (
        main(["--data-dir", str(data_dir), "--output", "json", "dataset", "list"])
        == 0
    )
    stdout = capsys.readouterr().out
    assert json.loads(stdout) == [{"name": "ds", "staged_images": 0}]

    assert main(["--data-dir", str(data_dir), "dataset", "create", "ds"]) == 1
    stderr = capsys.readouterr().err
    assert "Dataset already exists" in stderr


def test_cli_accepts_data_dir_at_command_levels(tmp_path, capsys):
    first = tmp_path / "first" / "data"
    second = tmp_path / "second" / "data"
    third = tmp_path / "third" / "data"

    assert main(["dataset", "--data-dir", str(first), "create", "ds"]) == 0
    assert (first / "staged" / "ds").exists()

    assert main(["dataset", "create", "ds", "--data-dir", str(second)]) == 0
    assert (second / "staged" / "ds").exists()

    assert main(["dataset", "list", "--data-dir", str(third)]) == 0
    assert "STAGED IMAGES" in capsys.readouterr().out
    assert third.exists()


def test_cli_import_extracts_trailing_data_dir_from_importer_args(tmp_path, monkeypatch):
    data_dir = tmp_path / "import-data"
    monkeypatch.setattr(six2one.Path, "home", lambda: tmp_path)

    def fake_run(command, *, check):
        out_dir = Path(command[-1])
        _image(out_dir / "0001.png")
        _six2one_metadata(out_dir / "0001.json", post_id=1)

    monkeypatch.setattr(six2one.subprocess, "run", fake_run)

    assert main(["candidates", "import", "621", "fox", "--data-dir", str(data_dir)]) == 0

    assert (data_dir / "candidates" / "0001.png").exists()
    assert (data_dir / "candidates" / "0001.json").exists()


def test_cli_candidates_caption_uses_command_options(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    paths = Paths(data_dir)
    _candidate(paths, "0001", tags=["existing"])

    def fake_build_tagger(*, presets):
        assert presets == ["natural_language"]
        return FakeTagger(["new"])

    monkeypatch.setattr(candidates_module, "build_tagger", fake_build_tagger)

    assert (
        main(
            [
                "candidates",
                "caption",
                "--data-dir",
                str(data_dir),
                "--all",
                "--preset",
                "natural_language",
                "--limit",
                "1",
                "--quiet",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "captioned: 1" in output
    assert json.loads((paths.candidates / "0001.json").read_text(encoding="utf-8"))[
        "tags"
    ] == ["existing", "new"]


def test_cli_candidates_caption_quiet_disables_progress(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    paths = Paths(data_dir)
    _candidate(paths, "0001", tags=["existing"])
    captured: dict[str, object] = {}

    def fake_caption(self, *, all_images, presets, limit, quiet):
        captured.update(
            {
                "all_images": all_images,
                "presets": presets,
                "limit": limit,
                "quiet": quiet,
            }
        )
        return []

    monkeypatch.setattr(Project(data_dir).candidates.__class__, "caption", fake_caption)

    assert (
        main(
            [
                "candidates",
                "caption",
                "--data-dir",
                str(data_dir),
                "--all",
                "--preset",
                "natural_language",
                "--limit",
                "1",
                "--quiet",
            ]
        )
        == 0
    )

    assert captured == {
        "all_images": True,
        "presets": ["natural_language"],
        "limit": 1,
        "quiet": True,
    }


def test_cli_dataset_stage_all(tmp_path, capsys):
    data_dir = tmp_path / "data"
    paths = Paths(data_dir)
    _candidate(paths, "0001")
    _candidate(paths, "0002")
    project = Project(paths.root)
    project.datasets.create("ds")

    assert main(["--data-dir", str(data_dir), "dataset", "stage", "ds", "--all"]) == 0
    output = capsys.readouterr().out

    assert "0001.png" in output
    assert "0002.png" in output
    assert project.datasets.status("ds").staged_images == 2


def test_cli_text_output_matches_architecture_shapes(tmp_path, capsys):
    data_dir = tmp_path / "data"
    paths = Paths(data_dir)
    _candidate(paths, "0001", tags=["fox"])
    _image(paths.candidates / "broken.png")
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "0001")
    (paths.models / "sd15.safetensors").write_bytes(b"model")

    assert main(["--data-dir", str(data_dir), "candidates", "list"]) == 0
    candidates_output = capsys.readouterr().out
    assert "# Candidates: 2" in candidates_output
    assert "# Broken:     1" in candidates_output
    assert "HAS_JSON" in candidates_output

    assert main(["--data-dir", str(data_dir), "dataset", "list"]) == 0
    assert "STAGED IMAGES" in capsys.readouterr().out

    assert main(["--data-dir", str(data_dir), "dataset", "status", "ds"]) == 0
    status_output = capsys.readouterr().out
    assert "Dataset: ds" in status_output
    assert "Prepared:         no" in status_output

    assert main(["--data-dir", str(data_dir), "models", "list"]) == 0
    models_output = capsys.readouterr().out
    assert "NAME" in models_output
    assert "TYPE" in models_output
    assert "FORMAT" in models_output
    assert "PATH" in models_output


def test_cli_empty_text_lists_use_command_specific_headers(tmp_path, capsys):
    data_dir = tmp_path / "data"

    assert main(["--data-dir", str(data_dir), "candidates", "list"]) == 0
    assert "# Candidates: 0" in capsys.readouterr().out

    assert main(["--data-dir", str(data_dir), "dataset", "list"]) == 0
    assert "STAGED IMAGES" in capsys.readouterr().out

    assert main(["--data-dir", str(data_dir), "models", "list"]) == 0
    models_output = capsys.readouterr().out
    assert "NAME" in models_output
    assert "TYPE" in models_output
    assert "FORMAT" in models_output
    assert "PATH" in models_output
    assert "# Candidates" not in models_output


def test_cli_table_output_keeps_columns_separated_for_long_names(tmp_path, capsys):
    data_dir = tmp_path / "data"
    paths = Paths(data_dir)
    project = Project(paths.root)
    project.datasets.create("long-dataset-name")
    (paths.models / "diffusers-demo").mkdir()
    (paths.models / "diffusers-demo" / "model_index.json").write_text("{}", encoding="utf-8")

    assert main(["--data-dir", str(data_dir), "dataset", "list"]) == 0
    assert "long-dataset-name  0" in capsys.readouterr().out

    assert main(["--data-dir", str(data_dir), "models", "list"]) == 0
    assert "diffusers-demo  diffusers" in capsys.readouterr().out


def test_cli_rejects_copy_mode_dimensions_from_manual_regression(tmp_path, capsys):
    data_dir = tmp_path / "data"
    paths = Paths(data_dir)
    _candidate(paths, "0001")
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "0001")

    code = main(
        [
            "--data-dir",
            str(data_dir),
            "dataset",
            "prepare",
            "ds",
            "--mode",
            "copy",
            "--width",
            "32",
        ]
    )

    assert code == 1
    assert "copy mode does not accept" in capsys.readouterr().err


def test_cli_copy_mode_without_dimensions_preserves_original_size(tmp_path):
    data_dir = tmp_path / "data"
    paths = Paths(data_dir)
    _candidate(paths, "0001", size=(64, 32))
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "0001")

    code = main(
        [
            "--data-dir",
            str(data_dir),
            "dataset",
            "prepare",
            "ds",
            "--mode",
            "copy",
        ]
    )

    assert code == 0
    assert _image_size(paths.prepared_for("ds") / "images" / "0001.png") == (64, 32)


def test_cli_prepare_accepts_no_remove_watermarks(tmp_path):
    data_dir = tmp_path / "data"
    paths = Paths(data_dir)
    _candidate(paths, "0001", size=(64, 32))
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "0001")

    code = main(
        [
            "--data-dir",
            str(data_dir),
            "dataset",
            "prepare",
            "ds",
            "--no-remove-watermarks",
        ]
    )

    assert code == 0
    config = json.loads((paths.prepared_for("ds") / "config.json").read_text(encoding="utf-8"))
    assert config["remove_watermarks"] is False


def test_cli_train_dry_run_prints_plan(tmp_path, capsys):
    data_dir = tmp_path / "data"
    paths = Paths(data_dir)
    _candidate(paths, "0001")
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "0001")
    (paths.models / "sd15.safetensors").write_bytes(b"model")

    code = main(
        [
            "--data-dir",
            str(data_dir),
            "--output",
            "json",
            "train",
            "ds",
            "--model",
            "sd15",
            "--dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["dry_run"] is True
    assert payload["plan"]["prepare_would_run"] is True


def test_cli_train_preset_values_and_overrides(tmp_path, capsys):
    data_dir = tmp_path / "data"
    paths = Paths(data_dir)
    _candidate(paths, "0001")
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "0001")
    (paths.models / "sd15.safetensors").write_bytes(b"model")

    code = main(
        [
            "--data-dir",
            str(data_dir),
            "--output",
            "json",
            "train",
            "ds",
            "--preset",
            "style-soft",
            "--steps",
            "2400",
            "--dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["plan"]["train_config"]["rank"] == 16
    assert payload["plan"]["train_config"]["steps"] == 2400
    assert payload["plan"]["train_config"]["learning_rate"] == 5e-5
    assert payload["plan"]["train_config"]["batch_size"] == 1
    assert payload["plan"]["train_config"]["gradient_accumulation"] == 4


def test_cli_train_help_lists_presets(capsys):
    with pytest.raises(SystemExit) as error:
        main(["train", "--help"])
    output = capsys.readouterr().out

    assert error.value.code == 0
    assert "--preset" in output
    assert "concept" in output
    assert "style-strong" in output
    assert "individual train flags still override" in output


def test_cli_size_cannot_be_combined_with_width(tmp_path, capsys):
    data_dir = tmp_path / "data"
    paths = Paths(data_dir)
    _candidate(paths, "0001")
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "0001")

    code = main(
        [
            "--data-dir",
            str(data_dir),
            "dataset",
            "prepare",
            "ds",
            "--size",
            "64",
            "--width",
            "32",
        ]
    )

    assert code == 1
    assert "--size cannot be combined" in capsys.readouterr().err


def test_candidate_duplicate_images_are_reported(tmp_path):
    paths = Paths(tmp_path / "data")
    paths.ensure()
    _image(paths.candidates / "same.png")
    _image(paths.candidates / "same.jpg")
    _metadata(paths.candidates / "same.json")

    with pytest.raises(LorakitError, match="multiple images"):
        candidates_module.list_all(paths)


def test_unstage_ambiguous_stem_is_reported(tmp_path):
    paths = Paths(tmp_path / "data")
    project = Project(paths.root)
    project.datasets.create("ds")
    _image(paths.staged_for("ds") / "same.png")
    _image(paths.staged_for("ds") / "same.jpg")

    with pytest.raises(LorakitError, match="Multiple staged images"):
        project.datasets.unstage("ds", "same")


def test_prepared_status_false_for_broken_manifest_entries(tmp_path):
    paths = Paths(tmp_path / "data")
    project = Project(paths.root)
    project.datasets.create("ds")
    prepared_dir = paths.prepared_for("ds")
    prepared_dir.mkdir(parents=True)
    (prepared_dir / "config.json").write_text("{}", encoding="utf-8")
    (prepared_dir / MANIFEST_NAME).write_text('{"caption": "missing image"}\n', encoding="utf-8")

    assert project.datasets.status("ds").prepared is False


def test_prepared_status_false_for_wrong_manifest_entry_types(tmp_path):
    paths = Paths(tmp_path / "data")
    project = Project(paths.root)
    project.datasets.create("ds")
    prepared_dir = paths.prepared_for("ds")
    images_dir = prepared_dir / "images"
    images_dir.mkdir(parents=True)
    _image(images_dir / "0001.png")
    (prepared_dir / "config.json").write_text("{}", encoding="utf-8")
    (prepared_dir / MANIFEST_NAME).write_text(
        '{"image": "images/0001.png", "caption": ["not", "text"], "tags": "not-list"}\n',
        encoding="utf-8",
    )

    assert project.datasets.status("ds").prepared is False


def _candidate(
    paths: Paths,
    stem: str,
    *,
    size: tuple[int, int] = (32, 32),
    tags: list[str] | None = None,
) -> None:
    paths.ensure()
    _image(paths.candidates / f"{stem}.png", size=size)
    _metadata(paths.candidates / f"{stem}.json", tags=tags)


def _image(
    path: Path,
    *,
    size: tuple[int, int] = (32, 32),
    mode: str = "RGB",
    color: object = (255, 0, 0),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.new(mode, size, color) as image:
        image.save(path)


def _metadata(path: Path, *, tags: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tags": tags if tags is not None else ["tag"], "metadata": {"source": "test"}}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_prepared_manifest(prepared_dir: Path, rows: list[dict[str, object]]) -> None:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row) for row in rows)
    (prepared_dir / MANIFEST_NAME).write_text(f"{content}\n", encoding="utf-8")


def _six2one_metadata(
    path: Path,
    *,
    post_id: int,
    general_tags: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": post_id,
        "tags": {
            "artist": ["artist_name"],
            "general": general_tags if general_tags is not None else ["anthro", "solo"],
            "meta": ["hi_res"],
            "species": ["fox"],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _six2one_output_post(output_dir: Path, post_id: int) -> None:
    image_path = output_dir / "images" / "sample" / f"{post_id:012d}.png"
    metadata_path = output_dir / "json" / f"{post_id:012d}.json"
    _image(image_path)
    _six2one_metadata(metadata_path, post_id=post_id)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": 3,
        "tool": {"name": "six2one", "version": "0.1.2"},
        "sources": {"e621": {"base_url": "https://e621.net"}},
        "output": {"root": str(output_dir), "root_absolute": str(output_dir.resolve())},
        "queries": {
            "e621:fox": {
                "key": "e621:fox",
                "compiled": "fox",
                "downloaded_count": 1,
                "last_post_id": post_id,
                "seen_post_ids": [post_id],
            },
        },
        "posts": {
            str(post_id): {
                "id": str(post_id),
                "file_paths": {
                    "json": f"json/{post_id:012d}.json",
                    "image_paths": {
                        "preview": None,
                        "sample": f"images/sample/{post_id:012d}.png",
                        "original": None,
                    },
                },
            },
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


class FakeTagger:
    def __init__(self, tags):
        self._tags = tags
        self.contexts = []

    def tags_for(self, image_path, *, context_tags=None):
        self.contexts.append(list(context_tags or []))
        return list(self._tags)


class FakeStructuredTagger:
    def __init__(self, result):
        self._result = result

    def results_for_many(self, requests):
        return [self._result for _ in requests]


class FakeIterativeStructuredTagger:
    def __init__(self, metadata_dir):
        self._metadata_dir = metadata_dir
        self.first_saved_before_second = False

    def results_for_many_iteratively(self, requests, *, quiet, on_result):
        results = []
        for index, request in enumerate(requests):
            if index == 1:
                self.first_saved_before_second = (
                    "tag-0"
                    in json.loads(
                        (self._metadata_dir / "0001.json").read_text(encoding="utf-8")
                    )["tags"]
                )
            result = tagging_module.TaggingResult(
                tags=[f"tag-{index}"],
                caption=f"Caption {index}.",
                sidecar={"index": index},
            )
            results.append(result)
            on_result(index, result)
        return results


class FakeTorchModule:
    class cuda:
        @staticmethod
        def is_available():
            return False

    class inference_mode:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False


class FakeQwenInputs(dict):
    def __init__(self):
        super().__init__(input_ids=[[1, 2]])
        self.input_ids = [[1, 2]]

    def to(self, device):
        self.device = device
        return self


class FakeQwenProcessor:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return "chat"

    def __call__(self, *, text, images, videos, padding, return_tensors):
        return FakeQwenInputs()

    def batch_decode(self, trimmed, *, skip_special_tokens, clean_up_tokenization_spaces):
        return ["{}"]


class FakeGenerateModel:
    def generate(self, **kwargs):
        return [[1, 2, 3]]


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size
