import json
from pathlib import Path

import pytest
import torch
from PIL import Image

import lorakit.models as models_module
import lorakit.training as training_module
from lorakit import candidates as candidates_module
from lorakit import PrepareConfig, Project, TrainingSpec
from lorakit.cli import main
from lorakit.errors import (
    DatasetExists,
    DatasetNotFound,
    ImporterMissing,
    InvalidPrepareConfig,
    ModelAmbiguous,
    LorakitError,
)
from lorakit.importers import six2one
from lorakit.manifest import MANIFEST_NAME, read_manifest
from lorakit.paths import Paths
from lorakit.training.backends.types import BackendResult


def test_project_creates_data_layout(tmp_path):
    project = Project(tmp_path / "data")

    assert project.paths.candidates.exists()
    assert project.paths.staged.exists()
    assert project.paths.prepared.exists()
    assert project.paths.models.exists()
    assert project.paths.models == tmp_path / "models"
    assert not (tmp_path / "data" / "models").exists()
    assert project.paths.artifacts.exists()


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


def test_candidates_tag_natural_uses_composite_tagger(tmp_path, monkeypatch):
    paths = Paths(tmp_path / "data")
    _image(paths.candidates / "0001.png")

    def fake_build_tagger(*, natural):
        assert natural is True
        return FakeTagger(["smilingwolf_tag", "natural language tag"])

    monkeypatch.setattr(candidates_module, "build_tagger", fake_build_tagger)

    results = Project(paths.root).candidates.tag(natural=True)

    assert results[0].added_tags == ["smilingwolf_tag", "natural language tag"]
    assert json.loads((paths.candidates / "0001.json").read_text(encoding="utf-8"))[
        "tags"
    ] == ["smilingwolf_tag", "natural language tag"]


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
            "caption": "lorakit, wide",
            "image": "images/wide.png",
            "tags": ["lorakit", "wide"],
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


def test_prepare_uses_staged_metadata_override(tmp_path):
    paths = Paths(tmp_path / "data")
    _candidate(paths, "item", tags=["candidate"])
    project = Project(paths.root)
    project.datasets.create("ds")
    project.datasets.stage("ds", "item")
    _metadata(paths.staged_for("ds") / "item.json", tags=["override"])

    prepared = project.datasets.prepare("ds", PrepareConfig(width=32, height=32))

    assert read_manifest(prepared / MANIFEST_NAME)[0]["caption"] == "override"


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
    assert rows[0]["caption"] == "hi_res, hi res, solo"
    assert rows[1]["tags"] == ["blue_eyes", "blue eyes"]
    assert rows[1]["caption"] == "blue_eyes, blue eyes"


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

    def fake_snapshot_download(*, repo_id, local_dir):
        local_dir.mkdir(parents=True)
        (local_dir / "model_index.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(models_module, "HfApi", FakeApi)
    monkeypatch.setattr(models_module, "snapshot_download", fake_snapshot_download)

    assert project.models.search("fox", limit=1)[0]["model_id"] == "owner/model"
    assert project.models.fetch("owner/new-model").name == "new-model"
    assert project.models.remove("pony").name == "pony.ckpt"


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


def test_importer_imports_pairs_skips_existing_and_reports_missing_binary(tmp_path, monkeypatch):
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
        "tags": ["anthro", "solo", "fox", "artist_name", "hi_res"],
    }

    second = six2one.import_pairs(paths, source)
    assert sorted(path.name for path in second.skipped) == ["0001.json", "0001.png"]

    monkeypatch.setattr(six2one.shutil, "which", lambda command: None)
    with pytest.raises(ImporterMissing):
        six2one.run(paths, ["fox"], overwrite=False)


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
    ] == ["new", "fox", "artist_name", "hi_res"]


def test_importer_run_shells_out_and_copies_results(tmp_path, monkeypatch):
    paths = Paths(tmp_path / "data")

    def fake_run(command, *, check):
        out_dir = Path(command[-1])
        _image(out_dir / "0001.png")
        _six2one_metadata(out_dir / "0001.json", post_id=1)

    monkeypatch.setattr(six2one.shutil, "which", lambda command: "/bin/621")
    monkeypatch.setattr(six2one.subprocess, "run", fake_run)

    result = six2one.run(paths, ["fox", "--safe"], overwrite=False)

    assert sorted(path.name for path in result.imported) == ["0001.json", "0001.png"]


def test_importer_run_preserves_site_in_candidate_metadata(tmp_path, monkeypatch):
    paths = Paths(tmp_path / "data")

    def fake_run(command, *, check):
        out_dir = Path(command[-1])
        _image(out_dir / "0001.png")
        _six2one_metadata(out_dir / "0001.json", post_id=1)

    monkeypatch.setattr(six2one.shutil, "which", lambda command: "/bin/621")
    monkeypatch.setattr(six2one.subprocess, "run", fake_run)

    six2one.run(paths, ["fox", "--site", "e926"], overwrite=False)

    metadata = json.loads((paths.candidates / "0001.json").read_text(encoding="utf-8"))
    assert metadata["metadata"]["source"] == "e926"


def test_importer_run_converts_command_failure_to_domain_error(tmp_path, monkeypatch):
    paths = Paths(tmp_path / "data")

    def fake_run(command, *, check):
        raise six2one.subprocess.CalledProcessError(returncode=2, cmd=command)

    monkeypatch.setattr(six2one.shutil, "which", lambda command: "/bin/621")
    monkeypatch.setattr(six2one.subprocess, "run", fake_run)

    with pytest.raises(LorakitError, match="six2one import failed"):
        six2one.run(paths, ["fox"], overwrite=False)


def test_cli_create_list_json_and_error_paths(tmp_path, capsys):
    data_dir = tmp_path / "data"
    assert main(["--data-dir", str(data_dir), "dataset", "create", "ds"]) == 0
    create_output = capsys.readouterr().out
    assert f"Created dataset ds at {data_dir / 'staged' / 'ds'}" in create_output
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

    def fake_run(command, *, check):
        out_dir = Path(command[-1])
        _image(out_dir / "0001.png")
        _six2one_metadata(out_dir / "0001.json", post_id=1)

    monkeypatch.setattr(six2one.shutil, "which", lambda command: "/bin/621")
    monkeypatch.setattr(six2one.subprocess, "run", fake_run)

    assert main(["candidates", "import", "621", "fox", "--data-dir", str(data_dir)]) == 0

    assert (data_dir / "candidates" / "0001.png").exists()
    assert (data_dir / "candidates" / "0001.json").exists()


def test_cli_candidates_tag_uses_command_options(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    paths = Paths(data_dir)
    _candidate(paths, "0001", tags=["existing"])

    def fake_build_tagger(*, natural):
        assert natural is True
        return FakeTagger(["new"])

    monkeypatch.setattr(candidates_module, "build_tagger", fake_build_tagger)

    assert (
        main(
            [
                "candidates",
                "tag",
                "--data-dir",
                str(data_dir),
                "--all",
                "--natural",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "Tagged:  1" in output
    assert json.loads((paths.candidates / "0001.json").read_text(encoding="utf-8"))[
        "tags"
    ] == ["existing", "new"]


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


class FakeTagger:
    def __init__(self, tags):
        self._tags = tags

    def tags_for(self, image_path):
        return list(self._tags)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size
