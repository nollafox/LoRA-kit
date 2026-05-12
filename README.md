# lorakit

<p align="center">
  <img src="docs/banner.png" alt="lorakit banner" style="border-radius: 16px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12); max-width: 100%; height: auto;">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11--3.12-1E90FF" alt="Python 3.11-3.12">
  <a href="https://github.com/nollafox/LoRA-kit/actions/workflows/test.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/nollafox/LoRA-kit/test.yml?branch=main&label=tests&color=2E8B57" alt="Test status">
  </a>
  <img src="https://img.shields.io/badge/CLI-lorakit-4169E1" alt="lorakit CLI">
  <img src="https://img.shields.io/badge/backend-diffusers-8A2BE2" alt="Diffusers backend">
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#how-lorakit-organizes-things">Data Layout</a> •
  <a href="#the-manifest">Manifest</a> •
  <a href="#commands">Commands</a>
</p>

**lorakit** is a command-line tool for building LoRA training datasets. You import source images, organize them into datasets by copying files into folders, and run one command to prepare and train. Each finished run is archived with the exact inputs that produced it.

The whole project is a directory you can browse with `tree` and edit with a text editor. There is no database, no hidden state, and nothing you can't see. If you've used git, the shape will feel familiar: a pool of available material, named units staged out of it, a build step, and durable artifacts at the end.

## Quick Start

Install lorakit from PyPI:

```bash
$ python -m pip install lorakit
```

That installs the `lorakit` command onto your `PATH`. A typical session imports source images, stages a curated subset, prepares them, and trains:

```bash
$ lorakit candidates import 621 fox solo --safe --limit 100
$ lorakit dataset create fox-solo
$ lorakit dataset stage fox-solo 000006394158
$ lorakit dataset stage fox-solo 000006394159
$ lorakit dataset prepare fox-solo --size 768 --trigger lorakit
$ lorakit train fox-solo --model sd15
```

The result is `data/artifacts/fox-solo/run-001/model.safetensors`, with a snapshot of the prepared inputs sitting next to it.

The `candidates import 621` step requires [six2one](https://github.com/nollafox/six2one), installable via `python -m pip install six2one`. `xformers` is optional; install `lorakit[accel]` on platforms where you want that acceleration and compatible wheels are available.

For an isolated install, use `pipx install lorakit`. For an editable install from a local clone, run `python -m pip install --user -e .`.

## How lorakit Organizes Things

Everything lorakit does happens inside `data/`. There are four working directories, each with a clear job:

```
data/
  candidates/                # The pool of source images + JSON pairs
    IMG_1.[ext]
    IMG_1.json
  staged/
    fox-solo/                # A dataset is a folder of staged images
      IMG_1.[ext]
      IMG_1.json             # Optional per-dataset tag override
  prepared/
    fox-solo/                # Derived from staged; rebuilt on every prepare
      images/
      lorakit-manifest.jsonl
      config.json
  models/                    # Base checkpoints available for training
  artifacts/
    fox-solo/
      run-001/               # A finished training run
        dataset/             # Immutable snapshot of the prepared inputs
          images/
          lorakit-manifest.jsonl
        model.safetensors
        config.json
```

**`candidates/`** is the pool of source material. Each candidate is an image plus a sibling JSON file with the same stem. Importers like `six2one` write here, and you can drop files in by hand whenever the pair stays matched.

The JSON shape is open-ended. `tags` and `metadata` are the conventional fields:

```json
{
  "tags": ["anthro fox", "male", "solo", "blue eyes"],
  "metadata": {
    "source": "e621",
    "post_id": 6394158
  }
}
```

A candidate is **valid** when both its image and JSON sibling exist. Missing one or the other makes it **broken**, which `candidates list` and `clean` will surface for you.

**`staged/<name>/`** is a dataset. Anything in the folder is part of that dataset — that's the whole rule. `lorakit dataset stage` copies files out of `candidates/` so the folder stays hand-editable, and `--symlink` is there if copies are wasteful.

A staged image uses the candidate's metadata by default. To override the tags for a specific dataset, drop a JSON file next to the staged image with the same stem.

**`prepared/<name>/`** is the training-ready build of a dataset. The preprocessed images live under `images/`, alongside a manifest describing them and the config used to build them. It's regenerated from scratch on every prepare or train, so treat it as disposable — never edit by hand, never depend on it across runs.

**`artifacts/<dataset>/run-NNN/`** is the durable record of a finished run. The trained LoRA, the config used, and a copied snapshot of the prepared inputs all sit together. Old runs stay reproducible even when the source dataset changes later, because every run carries its own inputs. Run numbers count up per dataset.

## The Manifest

Every prepared dataset comes with a `lorakit-manifest.jsonl`. One image per line, one JSON object per line:

```json
{"image": "images/IMG_1.png", "caption": "lorakit, anthro fox, male, solo, blue eyes", "tags": ["lorakit", "anthro fox", "male", "solo", "blue eyes"]}
```

Each row carries two views of the same information. `caption` is the comma-joined tag string that training backends actually read; if you pass `--trigger`, the trigger word is prepended here. `tags` keeps the structured list so downstream tools can use it without re-parsing.

Different backends adapt cheaply. A Diffusers run, for instance, rewrites each row into `{"file_name": "...", "text": "..."}` on the way in.

The manifest is rebuilt on every `dataset prepare`. The copy inside a finished run's `artifacts/<dataset>/run-NNN/dataset/` is the immutable record of what that run actually saw.

## Commands

### Importing candidates

```bash
$ lorakit candidates import 621 fox solo --safe --limit 100 [--overwrite]
```

`candidates import` is how source material enters the pool. The first positional argument names the importer, and the rest are passed straight to it. The only importer in this build is `621`, which wraps [six2one](https://github.com/nollafox/six2one) and accepts the same tag query you would type into e621's search bar.

The importer runs in a temporary directory, then its image-and-JSON pairs are copied into `data/candidates/`. Files that already exist with the same stem are skipped silently — pass `--overwrite` to replace them. If `six2one` is not on your `PATH`, lorakit exits with installation guidance.

Candidate stems are global within `data/candidates/`. The `621` importer uses e621 post IDs, which are already unique. Future importers from sources without natural unique IDs should prefix their stems (`local_IMG_0001`, say) to avoid collisions.

### Working with candidates

```bash
$ lorakit candidates list

# Candidates: 240
# Valid:      237
# Broken:     3
#
# NAME            IMAGE                         HAS_JSON
# 000006394158    candidates/000006394158.jpg   yes
# 000006394159    candidates/000006394159.png   yes
```

The summary line counts valid and broken candidates; the table lists them. When nothing is broken, the "Broken" line and broken rows drop out of `text` output, so a healthy pool stays quiet. `--output json` and `--output jsonl` include every row regardless.

```bash
$ lorakit candidates show 000006394158
```

`show` pretty-prints the candidate's JSON to stdout. Handy for grepping, scripting, or piping into another tool.

### Datasets

```bash
$ lorakit dataset create fox-solo
$ lorakit dataset stage fox-solo 000006394158
$ lorakit dataset stage fox-solo 000006394159 --symlink
$ lorakit dataset unstage fox-solo 000006394159
$ lorakit dataset list
$ lorakit dataset status fox-solo
$ lorakit dataset rename fox-solo fox-solo-v2
$ lorakit dataset delete fox-solo-v2 [--yes]
```

`stage` copies an image from `candidates/` into the dataset folder. The image keeps its candidate metadata by default. To override the tags for this dataset, drop a JSON file next to it with the same stem.

`--symlink` substitutes a symlink for the copy. That saves space, at the cost of hand-editability — the linked file still lives back in `candidates/`.

`unstage` removes a single image from the dataset. `delete` removes the entire folder. Neither touches `artifacts/`, so previously trained models stay reproducible even when their source dataset is gone. In an interactive shell, `delete` prompts before deleting; in scripts, pass `--yes`.

`status` prints a summary that includes whether the dataset is currently prepared:

```
Dataset: fox-solo
Staged images:    42
Overrides:        5
Missing metadata: 0
Prepared:         no
Artifacts:        2
```

`Prepared: yes` requires `prepared/<name>/` to exist with both `config.json` and `lorakit-manifest.jsonl` and no broken entries; anything less reads as `no`. `Artifacts` is the number of run directories under `artifacts/<name>/`.

### Preparing a dataset

```bash
$ lorakit dataset prepare fox-solo [options]
```

Preparation clears `prepared/<dataset>/` and rebuilds it from the staged images. The options control how source images are reshaped into training input:

| Option | Default | Description |
| --- | --- | --- |
| `--mode` | `fit` | `copy`, `fit`, `center-crop`, or `pad`. |
| `--size N` | `512` | Shortcut for `--width N --height N`. Cannot be combined with `--width`/`--height`. |
| `--width W` / `--height H` | from `--size` | Explicit target box. |
| `--image-format` | `original` | `png`, `jpg`, `webp`, or `original`. |
| `--trigger` | `""` | Trigger word prepended to every image's tags. |

The four modes differ in how they handle aspect ratio:

- `copy` does nothing. Images are copied unchanged; any explicit dimension flag is an error.
- `fit` resizes to fit inside the target box, preserving aspect ratio.
- `center-crop` crops to the target aspect ratio and then resizes to the target box.
- `pad` resizes to fit, then pads the remainder to fill the box.

If you supply only one of `--width` or `--height`, `fit` constrains that one dimension; `center-crop` and `pad` mirror it for the missing one; `copy` errors.

`--image-format original` passes the source format through. `png`, `jpg`, and `webp` convert. Converting a transparent source to `jpg` errors rather than silently flattening — handle transparency at the source, or use a format that supports it.

### Models

The `models` group manages the base checkpoints you train on top of.

```bash
$ lorakit models list

# NAME        TYPE        FORMAT       PATH
# sd15        checkpoint  safetensors  data/models/sd15.safetensors
# pony-v6     checkpoint  safetensors  data/models/pony-v6.safetensors
# sdxl-base   diffusers   directory    data/models/sdxl-base
```

Two formats are discovered automatically under `data/models/`:

- Single checkpoint files: `*.safetensors`, `*.ckpt`, `*.pt`.
- Diffusers model directories: any folder containing `model_index.json`.

```bash
$ lorakit models search sdxl [--task text-to-image] [--limit 20]
$ lorakit models fetch stable-diffusion-v1-5/stable-diffusion-v1-5 [--name sd15]
$ lorakit models remove pony-v6 [--yes]
```

`search` queries Hugging Face via `huggingface_hub.HfApi.list_models()` and respects its filters.

`fetch` downloads a Hugging Face repository snapshot into `data/models/<name>/`. `--name` defaults to the remote name. Private or gated repos use the standard `HF_TOKEN` environment variable or a prior `hf auth login`.

`remove` resolves the local name using the same rules as the `--model` flag — `data/models/<name>/`, then `data/models/<name>.*` — and refuses to act on an ambiguous match.

### Training

```bash
$ lorakit train fox-solo --model sd15
```

That is the common case: train the staged dataset `fox-solo` against the base model `sd15`, using sensible defaults for everything else. Every default is exposed as a flag for when you need them:

```bash
$ lorakit train fox-solo \
    --model sd15 \
    --backend diffusers \
    --resolution 512 \
    --rank 16 \
    --steps 2000 \
    --learning-rate 1e-4 \
    --batch-size 1 \
    --gradient-accumulation 4 \
    --mixed-precision fp16
```

`train` always runs `dataset prepare` first, so a single command goes from staged folder to trained LoRA. Any prepare flags you pass to `train` flow through to that step. Pass `--no-prepare` to skip preparation and reuse whatever is already in `prepared/<dataset>/`.

Either way, the prepared dataset is copied into `artifacts/<dataset>/run-NNN/dataset/` so the run stays reconstructable.

`--dry-run` resolves the model, prints the effective prepare and train configs, and exits — without touching files or launching training. It also reports whether prepare would have run.

`--run-name` overrides the default `run-NNN` directory name.

`--model` resolves in this order:

1. A path that exists on disk.
2. `data/models/<name>/` as a Diffusers directory.
3. `data/models/<name>.*` as a checkpoint file.
4. Otherwise, a Hugging Face repo ID.

The resolved path is recorded in the run's `config.json` so future audits know which model was used.

On success, the run lands at `data/artifacts/<dataset>/run-NNN/`, where NNN is one greater than the largest existing run for that dataset. The trainer's output file is normalized to `model.safetensors`.

The Diffusers backend tries to enable `xformers` memory-efficient attention when it is installed. If it is missing, training falls back to PyTorch attention. Install the optional accelerator extra only on platforms where `xformers` is supported:

```bash
$ python -m pip install 'lorakit[accel]'
```

### Cleaning data

```bash
$ lorakit clean [--apply]
```

`clean` scans `candidates/` and `staged/` for orphans — files whose siblings are missing — and proposes deletions. Run it with no flags to preview, or `--apply` to perform the deletes.

The rules:

- In `candidates/`, an image without a JSON sibling, or a JSON without an image sibling, is an orphan.
- In `staged/<dataset>/`, a staged image is an orphan when it has no candidate JSON and no staged override.
- A staged JSON is an orphan when there's no matching staged image.

Orphans never block other commands. They simply surface as broken in `candidates list` and as missing metadata in `dataset status`.

## Global Options

Two flags apply across the CLI:

| Flag | Where | Description |
| --- | --- | --- |
| `--data-dir <path>` | Every command. | Overrides the default `./data` location. Useful for tests and for keeping multiple projects in one install. |
| `--output text\|json\|jsonl` | Commands that print structured results. | Defaults to `text`. Destructive commands ignore it. |

There is no `lorakit init`. Commands auto-create any missing pieces of the data directory on first use.

## Command Reference

```
lorakit candidates import 621 ... [--overwrite]
lorakit candidates list
lorakit candidates show <stem>
lorakit clean [--apply]
lorakit dataset create <name>
lorakit dataset delete <name> [--yes]
lorakit dataset rename <old> <new>
lorakit dataset list
lorakit dataset status <name>
lorakit dataset stage <dataset> <image> [--symlink]
lorakit dataset unstage <dataset> <image>
lorakit dataset prepare <dataset> [options]
lorakit models list
lorakit models search <query>
lorakit models fetch <repo_id> [--name NAME]
lorakit models remove <name> [--yes]
lorakit train <dataset> [options]
```

## Development

Install with Poetry:

```bash
$ poetry install
```

Run the CLI and the test suite from inside the Poetry environment:

```bash
$ poetry run lorakit --help
$ poetry run lorakit dataset list
$ poetry run pytest
$ poetry run python -m compileall -q src
```

The CLI is a thin layer over a standalone core. You can import the domain modules — `candidates`, `datasets`, `prepare`, `models`, `training`, `clean` — and call them directly. A `Project` facade at `lorakit.Project` wraps them in the same grouped namespace the CLI uses:

```python
from lorakit import Project, PrepareConfig, TrainingSpec

p = Project("./data")
p.candidates.import_from("621", ["fox", "solo", "--safe", "--limit", "100"])
p.datasets.create("fox-solo")
p.datasets.stage("fox-solo", "000006394158")
p.datasets.prepare("fox-solo", PrepareConfig(mode="center-crop", width=768))
p.train(TrainingSpec(dataset="fox-solo", model="sd15"))
```

A future HTTP API or UI calls the same surface.

<br>

***

<p align="center">
  <strong>lorakit</strong> — git-style project management for LoRA training.
</p>

<p align="center">
  Crafted by <strong>Nolla Fox</strong>
</p>
