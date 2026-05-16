"""Diffusers LoRA training backend.

This backend is intentionally zero-config from the CLI side.  Internally it uses
heldout SNR-bucket validation to select checkpoints and to activate quality
challengers only when their virtual one-step validation score improves beyond a
scale-aware tolerance.

Key non-heuristic invariants:
- Latent cache is deterministic: posterior mode when available, otherwise mean.
- LoRA+ is implemented as actual AdamW param-group learning-rate ratios, not by
  scaling gradients before Adam's moment normalization.
- Objective / LoRA+ challenger comparisons use the same frozen training
  perturbation: identical batch, timesteps, and noise.
- Rank growth is triggered by validation plateau and uses dense base-weight
  validation gradients split into signal/noise estimates.  New rank channels are
  no-op initialized so model outputs are unchanged at insertion time.
"""

from __future__ import annotations

import gc
import shutil
from dataclasses import dataclass, replace
from importlib.util import find_spec
from pathlib import Path
from typing import Final, Protocol

import torch
from accelerate import Accelerator
from peft import LoraConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler

from lorakit.errors import LorakitError
from lorakit.training.backends._artifacts import (
    save_named_lora_weights,
    write_policy_challenger_report_if_main,
    write_probe_summary,
)
from lorakit.training.backends._cache import (
    CACHE_DIR_NAME,
    CACHE_MANIFEST_NAME,
    CacheRecord,
    CachedBatch,
    DiskCachedLatentDataset,
    LatentShapeBatchSampler,
    TrainingRow,
    cache_encoder_hidden_states,
    cache_latents,
    cache_records,
    cleanup_after_cuda_oom,
    collate_cached,
    is_cuda_oom,
    load_rows,
    write_cache_manifest,
)
from lorakit.training.backends._challengers import FrozenBatch, PolicyChallenger
from lorakit.training.backends._evaluator import DiffusionValidator
from lorakit.training.backends._loss import LossContext, loss_context
from lorakit.training.backends._optimizer import (
    OptimizerConfig,
    build_lora_optimizer,
    set_lora_plus_optimizer_ratio,
)
from lorakit.training.backends._policy import TrainingPolicy, default_training_policy
from lorakit.training.backends._probes import (
    ContextProber,
    GuardrailDecision,
    ProbeConfig,
    default_guardrail_decision,
    should_probe_contexts,
    write_probe_status_if_main,
)
from lorakit.training.backends._rank import maybe_grow_lora_rank
from lorakit.training.backends._validation import (
    BestCheckpoint,
    ValidationSkeleton,
    build_validation_skeleton,
    split_train_validation_records,
    write_validation_report_if_main,
)
from lorakit.training.backends.types import BackendResult, BackendSpec


LORA_TARGET_MODULES: Final = ["to_k", "to_q", "to_v", "to_out.0"]
LORA_WEIGHTS_NAME: Final = "pytorch_lora_weights.safetensors"
BEST_LORA_WEIGHTS_NAME: Final = "best_lora_weights.safetensors"
MAX_GRAD_NORM: Final = 1.0
LR_WARMUP_STEPS: Final = 0

VALIDATION_EVERY_PROBES: Final = 1

PROBE_LOG_NAME: Final = "context-probes.jsonl"
PROBE_SUMMARY_NAME: Final = "context-probes-summary.json"


class _OptimizerLike(Protocol):
    param_groups: list[dict[str, object]]

    def step(self) -> None:
        ...

    def zero_grad(self, *, set_to_none: bool = False) -> None:
        ...


class _SchedulerLike(Protocol):
    def step(self) -> None:
        ...


@dataclass(frozen=True)
class _BackendPaths:
    output_dir: Path

    @classmethod
    def create(cls, output_dir: Path) -> "_BackendPaths":
        output_dir.mkdir(parents=True, exist_ok=True)
        return cls(output_dir=output_dir)

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / CACHE_DIR_NAME

    @property
    def cache_manifest_path(self) -> Path:
        return self.cache_dir / CACHE_MANIFEST_NAME

    @property
    def probe_log_path(self) -> Path:
        return self.output_dir / PROBE_LOG_NAME

    @property
    def probe_summary_path(self) -> Path:
        return self.output_dir / PROBE_SUMMARY_NAME

    @property
    def final_weights_path(self) -> Path:
        return self.output_dir / LORA_WEIGHTS_NAME

    @property
    def best_weights_path(self) -> Path:
        return self.output_dir / BEST_LORA_WEIGHTS_NAME

    def artifact_paths(self) -> tuple[Path, ...]:
        return tuple(
            path
            for path in (self.probe_log_path, self.probe_summary_path, self.best_weights_path)
            if path.exists()
        )


@dataclass(frozen=True)
class _LoadedComponents:
    noise_scheduler: DDPMScheduler
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel

    @classmethod
    def load(cls, model: str) -> "_LoadedComponents":
        model_path = Path(model)
        if model_path.exists() and model_path.is_file():
            return cls.from_single_file(model_path)

        model_ref = str(model_path) if model_path.exists() else model
        return cls(
            noise_scheduler=DDPMScheduler.from_pretrained(model_ref, subfolder="scheduler"),
            tokenizer=CLIPTokenizer.from_pretrained(model_ref, subfolder="tokenizer"),
            text_encoder=CLIPTextModel.from_pretrained(
                model_ref,
                subfolder="text_encoder",
                low_cpu_mem_usage=False,
            ),
            vae=AutoencoderKL.from_pretrained(
                model_ref,
                subfolder="vae",
                low_cpu_mem_usage=False,
            ),
            unet=UNet2DConditionModel.from_pretrained(
                model_ref,
                subfolder="unet",
                low_cpu_mem_usage=False,
            ),
        )

    @classmethod
    def from_single_file(cls, model_path: Path) -> "_LoadedComponents":
        pipeline = StableDiffusionPipeline.from_single_file(
            str(model_path),
            torch_dtype=torch.float32,
            safety_checker=None,
        )
        return cls(
            noise_scheduler=DDPMScheduler.from_config(pipeline.scheduler.config),
            tokenizer=pipeline.tokenizer,
            text_encoder=pipeline.text_encoder,
            vae=pipeline.vae,
            unet=pipeline.unet,
        )

    def trainable_components(self) -> "_TrainableComponents":
        return _TrainableComponents(
            noise_scheduler=self.noise_scheduler,
            unet=self.unet,
        )


@dataclass(frozen=True)
class _TrainableComponents:
    noise_scheduler: DDPMScheduler
    unet: UNet2DConditionModel


@dataclass(frozen=True)
class _TensorCache:
    records: list[CacheRecord]

    @classmethod
    def create_on_disk(
        cls,
        *,
        rows: list[TrainingRow],
        paths: _BackendPaths,
        components: _LoadedComponents,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "_TensorCache":
        encoder_hidden_state_paths = cache_encoder_hidden_states(
            rows=rows,
            cache_dir=paths.cache_dir,
            tokenizer=components.tokenizer,
            text_encoder=components.text_encoder,
            device=device,
            dtype=dtype,
        )
        latent_paths = cache_latents(
            rows=rows,
            cache_dir=paths.cache_dir,
            vae=components.vae,
            device=device,
            dtype=dtype,
        )
        records = cache_records(
            rows=rows,
            latent_paths=latent_paths,
            encoder_hidden_state_paths=encoder_hidden_state_paths,
        )
        latent_scaling_factor = float(components.vae.config.scaling_factor)
        write_cache_manifest(
            paths.cache_manifest_path,
            records,
            latent_scaling_factor=latent_scaling_factor,
        )
        return cls(records=records)


@dataclass(frozen=True)
class _TrainingWorkload:
    dataset: DiskCachedLatentDataset
    dataloader: DataLoader
    validator: DiffusionValidator
    validation_skeleton: ValidationSkeleton

    @classmethod
    def from_cache(
        cls,
        *,
        records: list[CacheRecord],
        batch_size: int,
        unet: UNet2DConditionModel,
        noise_scheduler: DDPMScheduler,
        weight_dtype: torch.dtype,
    ) -> "_TrainingWorkload":
        train_records, validation_records = split_train_validation_records(records)
        validation_skeleton = build_validation_skeleton(
            records=validation_records,
            noise_scheduler=noise_scheduler,
        )
        validator = DiffusionValidator(
            skeleton=validation_skeleton,
            unet=unet,
            noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
        )
        dataset = DiskCachedLatentDataset(records=train_records)
        dataloader = DataLoader(
            dataset,
            batch_sampler=LatentShapeBatchSampler(records=train_records, batch_size=batch_size),
            collate_fn=collate_cached,
            num_workers=0,
            pin_memory=False,
        )
        return cls(
            dataset=dataset,
            dataloader=dataloader,
            validator=validator,
            validation_skeleton=validation_skeleton,
        )

    def with_prepared_dataloader(self, dataloader: DataLoader) -> "_TrainingWorkload":
        return replace(self, dataloader=dataloader)


@dataclass
class _TrainingState:
    global_step: int
    validation_probe_count: int
    policy: TrainingPolicy
    best_checkpoint: BestCheckpoint

    @classmethod
    def start(cls, *, best_checkpoint_path: Path) -> "_TrainingState":
        return cls(
            global_step=0,
            validation_probe_count=0,
            policy=default_training_policy(),
            best_checkpoint=BestCheckpoint.empty(best_checkpoint_path),
        )

    def record_validation_probe(self) -> bool:
        self.validation_probe_count += 1
        return self.validation_probe_count % VALIDATION_EVERY_PROBES == 0


@dataclass(frozen=True)
class _TrainingRuntime:
    accelerator: Accelerator
    spec: BackendSpec
    paths: _BackendPaths
    unet: torch.nn.Module
    optimizer: _OptimizerLike
    scheduler: _SchedulerLike
    workload: _TrainingWorkload
    prober: ContextProber
    probe_config: ProbeConfig
    noise_scheduler: DDPMScheduler
    weight_dtype: torch.dtype


def train(spec: BackendSpec) -> BackendResult:
    _validate_spec(spec)

    paths = _BackendPaths.create(spec.output_dir)
    rows = load_rows(spec.prepared_dir)
    accelerator = _accelerator_for(spec)

    components = _LoadedComponents.load(spec.model)
    attention_backend = _install_trainable_lora_adapter(
        components=components,
        rank=spec.rank,
    )
    weight_dtype = _weight_dtype(accelerator)

    tensor_cache = _TensorCache.create_on_disk(
        rows=rows,
        paths=paths,
        components=components,
        device=accelerator.device,
        dtype=weight_dtype,
    )
    trainable_components = components.trainable_components()
    _offload_cache_only_components(text_encoder=components.text_encoder, vae=components.vae)
    del components

    _prepare_unet_for_training(
        unet=trainable_components.unet,
        accelerator=accelerator,
        weight_dtype=weight_dtype,
    )
    workload = _TrainingWorkload.from_cache(
        records=tensor_cache.records,
        batch_size=spec.batch_size,
        unet=trainable_components.unet,
        noise_scheduler=trainable_components.noise_scheduler,
        weight_dtype=weight_dtype,
    )
    runtime = _prepare_training_runtime(
        spec=spec,
        paths=paths,
        accelerator=accelerator,
        trainable_components=trainable_components,
        workload=workload,
        weight_dtype=weight_dtype,
    )
    state = _TrainingState.start(best_checkpoint_path=paths.best_weights_path)

    _run_training_loop(
        runtime=runtime,
        state=state,
        attention_backend=attention_backend,
    )
    _write_final_lora_weights(runtime=runtime)
    return _backend_result(paths=paths)


def _accelerator_for(spec: BackendSpec) -> Accelerator:
    return Accelerator(
        gradient_accumulation_steps=spec.gradient_accumulation,
        mixed_precision=_accelerate_mixed_precision(spec.mixed_precision),
        project_dir=spec.output_dir,
    )


def _install_trainable_lora_adapter(
    *,
    components: _LoadedComponents,
    rank: int,
) -> str:
    components.unet.requires_grad_(False)
    components.vae.requires_grad_(False)
    components.text_encoder.requires_grad_(False)
    components.unet.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=LORA_TARGET_MODULES,
        )
    )
    components.unet.enable_gradient_checkpointing()
    return _enable_memory_efficient_attention(components.unet)


def _offload_cache_only_components(
    *,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
) -> None:
    text_encoder.to("cpu")
    vae.to("cpu")
    _release_cuda_cache()


def _prepare_unet_for_training(
    *,
    unet: UNet2DConditionModel,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
) -> None:
    unet.to(accelerator.device, dtype=weight_dtype)
    if accelerator.mixed_precision == "fp16":
        _cast_trainable_parameters_to_fp32(unet)
        _assert_trainable_parameters_are_fp32(unet)


def _prepare_training_runtime(
    *,
    spec: BackendSpec,
    paths: _BackendPaths,
    accelerator: Accelerator,
    trainable_components: _TrainableComponents,
    workload: _TrainingWorkload,
    weight_dtype: torch.dtype,
) -> _TrainingRuntime:
    optimizer = build_lora_optimizer(
        trainable_components.unet,
        OptimizerConfig(learning_rate=spec.learning_rate),
    )
    scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=LR_WARMUP_STEPS,
        num_training_steps=spec.steps * accelerator.num_processes,
    )
    unet, optimizer, dataloader, scheduler = accelerator.prepare(
        trainable_components.unet,
        optimizer,
        workload.dataloader,
        scheduler,
    )
    prepared_workload = workload.with_prepared_dataloader(dataloader)
    probe_config = ProbeConfig()
    prober = ContextProber(
        unet=unet,
        optimizer=optimizer,
        dataset=prepared_workload.dataset,
        noise_scheduler=trainable_components.noise_scheduler,
        weight_dtype=weight_dtype,
        config=probe_config,
        cuda_memory_snapshot=_cuda_memory_snapshot,
    )
    return _TrainingRuntime(
        accelerator=accelerator,
        spec=spec,
        paths=paths,
        unet=unet,
        optimizer=optimizer,
        scheduler=scheduler,
        workload=prepared_workload,
        prober=prober,
        probe_config=probe_config,
        noise_scheduler=trainable_components.noise_scheduler,
        weight_dtype=weight_dtype,
    )


def _run_training_loop(
    *,
    runtime: _TrainingRuntime,
    state: _TrainingState,
    attention_backend: str,
) -> None:
    progress = tqdm(
        range(runtime.spec.steps),
        desc="Steps",
        disable=not runtime.accelerator.is_local_main_process,
    )
    if runtime.accelerator.is_local_main_process:
        tqdm.write(f"Attention backend: {attention_backend}")

    while state.global_step < runtime.spec.steps:
        for batch in runtime.workload.dataloader:
            loss = _train_batch(runtime=runtime, state=state, batch=batch)

            if runtime.accelerator.sync_gradients:
                state.global_step += 1
                progress.update(1)
                progress.set_postfix(loss=float(loss.detach().item()))
                if state.global_step >= runtime.spec.steps:
                    break

    progress.close()


def _train_batch(
    *,
    runtime: _TrainingRuntime,
    state: _TrainingState,
    batch: CachedBatch,
) -> torch.Tensor:
    with runtime.accelerator.accumulate(runtime.unet):
        set_lora_plus_optimizer_ratio(
            runtime.optimizer,
            base_learning_rate=runtime.spec.learning_rate,
            ratio=state.policy.lora_plus_ratio,
        )
        current_loss_context = loss_context(
            batch=batch,
            unet=runtime.unet,
            noise_scheduler=runtime.noise_scheduler,
            weight_dtype=runtime.weight_dtype,
            objective=state.policy.objective,
        )
        loss = current_loss_context.loss
        runtime.accelerator.backward(loss)

        if runtime.accelerator.sync_gradients:
            runtime.accelerator.clip_grad_norm_(
                [parameter for parameter in runtime.unet.parameters() if parameter.requires_grad],
                MAX_GRAD_NORM,
            )

        should_probe, guardrail_decision = _probe_update_guardrail(
            runtime=runtime,
            state=state,
            batch=batch,
            current_loss_context=current_loss_context,
            loss=loss,
        )
        committed_update = _apply_update(
            runtime=runtime,
            policy=state.policy,
            guardrail_decision=guardrail_decision,
        )
        _maybe_validate_and_update_policy(
            runtime=runtime,
            state=state,
            batch=batch,
            current_loss_context=current_loss_context,
            should_probe=should_probe,
            committed_update=committed_update,
        )
        runtime.optimizer.zero_grad(set_to_none=True)
        return loss


def _probe_update_guardrail(
    *,
    runtime: _TrainingRuntime,
    state: _TrainingState,
    batch: CachedBatch,
    current_loss_context: LossContext,
    loss: torch.Tensor,
) -> tuple[bool, GuardrailDecision]:
    should_probe = should_probe_contexts(
        global_step=state.global_step,
        sync_gradients=runtime.accelerator.sync_gradients,
        free_cuda_bytes=_cuda_free_bytes(),
        config=runtime.probe_config,
    )
    if should_probe:
        return True, _run_context_probe(
            runtime=runtime,
            state=state,
            batch=batch,
            current_loss_context=current_loss_context,
            loss=loss,
        )

    _write_low_memory_probe_skip_if_needed(runtime=runtime, state=state)
    return False, default_guardrail_decision()


def _run_context_probe(
    *,
    runtime: _TrainingRuntime,
    state: _TrainingState,
    batch: CachedBatch,
    current_loss_context: LossContext,
    loss: torch.Tensor,
) -> GuardrailDecision:
    cuda_memory_before_probe = _cuda_memory_snapshot()
    _write_probe_started(
        runtime=runtime,
        state=state,
        current_loss_context=current_loss_context,
        cuda_memory_before_probe=cuda_memory_before_probe,
    )
    try:
        return runtime.prober.run(
            batch=batch,
            noise=current_loss_context.noise,
            timesteps=current_loss_context.timesteps,
            probe_log_path=runtime.paths.probe_log_path,
            step=state.global_step,
            should_log=runtime.accelerator.is_local_main_process,
            requested_batch_size=runtime.spec.batch_size,
            gradient_accumulation=runtime.spec.gradient_accumulation,
            training_loss=loss,
            candidate_step_size=runtime.spec.learning_rate,
            cuda_memory_before_probe=cuda_memory_before_probe,
        )
    except RuntimeError as exc:
        if not is_cuda_oom(exc):
            raise
        cleanup_after_cuda_oom()
        _write_probe_cuda_oom(runtime=runtime, state=state)
        return default_guardrail_decision()


def _write_probe_started(
    *,
    runtime: _TrainingRuntime,
    state: _TrainingState,
    current_loss_context: LossContext,
    cuda_memory_before_probe: dict[str, int] | None,
) -> None:
    write_probe_status_if_main(
        probe_log_path=runtime.paths.probe_log_path,
        step=state.global_step,
        should_log=runtime.accelerator.is_local_main_process,
        status="started",
        free_cuda_bytes=(
            cuda_memory_before_probe["free_cuda_bytes"]
            if cuda_memory_before_probe is not None
            else None
        ),
        extra={
            "probe_version": runtime.probe_config.version,
            "microbatch_size": int(current_loss_context.noise.shape[0]),
            "requested_batch_size": int(runtime.spec.batch_size),
            "gradient_accumulation": int(runtime.spec.gradient_accumulation),
            "active_objective": state.policy.objective.label,
            "active_lora_plus_ratio": float(state.policy.lora_plus_ratio),
        },
    )


def _write_probe_cuda_oom(
    *,
    runtime: _TrainingRuntime,
    state: _TrainingState,
) -> None:
    write_probe_status_if_main(
        probe_log_path=runtime.paths.probe_log_path,
        step=state.global_step,
        should_log=runtime.accelerator.is_local_main_process,
        status="skipped_cuda_oom",
        free_cuda_bytes=_cuda_free_bytes(),
        extra={
            "probe_version": runtime.probe_config.version,
            "requested_batch_size": int(runtime.spec.batch_size),
            "gradient_accumulation": int(runtime.spec.gradient_accumulation),
        },
    )


def _write_low_memory_probe_skip_if_needed(
    *,
    runtime: _TrainingRuntime,
    state: _TrainingState,
) -> None:
    if not runtime.accelerator.sync_gradients or not runtime.accelerator.is_local_main_process:
        return
    free_cuda_bytes = _cuda_free_bytes()
    if free_cuda_bytes is None or free_cuda_bytes >= runtime.probe_config.min_free_cuda_bytes:
        return
    write_probe_status_if_main(
        probe_log_path=runtime.paths.probe_log_path,
        step=state.global_step,
        should_log=True,
        status="skipped_low_cuda_memory",
        free_cuda_bytes=free_cuda_bytes,
        extra={
            "probe_version": runtime.probe_config.version,
            "requested_batch_size": int(runtime.spec.batch_size),
            "gradient_accumulation": int(runtime.spec.gradient_accumulation),
        },
    )


def _apply_update(
    *,
    runtime: _TrainingRuntime,
    policy: TrainingPolicy,
    guardrail_decision: GuardrailDecision,
) -> bool:
    if not guardrail_decision.handled_update:
        set_lora_plus_optimizer_ratio(
            runtime.optimizer,
            base_learning_rate=runtime.spec.learning_rate,
            ratio=policy.lora_plus_ratio,
        )
        runtime.optimizer.step()

    committed_update = not guardrail_decision.skipped_update
    if committed_update:
        runtime.scheduler.step()
    return committed_update


def _maybe_validate_and_update_policy(
    *,
    runtime: _TrainingRuntime,
    state: _TrainingState,
    batch: CachedBatch,
    current_loss_context: LossContext,
    should_probe: bool,
    committed_update: bool,
) -> None:
    if not should_probe or not committed_update or not runtime.workload.validation_skeleton.items:
        return
    if not state.record_validation_probe():
        return

    validation_report = runtime.workload.validator.score(step=state.global_step)
    improved, state.best_checkpoint = state.best_checkpoint.consider(validation_report)
    if improved and runtime.accelerator.is_main_process:
        save_named_lora_weights(
            unet=runtime.accelerator.unwrap_model(runtime.unet),
            output_dir=runtime.spec.output_dir,
            filename=BEST_LORA_WEIGHTS_NAME,
        )
    write_validation_report_if_main(
        probe_log_path=runtime.paths.probe_log_path,
        report=validation_report,
        best_checkpoint=state.best_checkpoint,
        improved=improved,
        should_log=runtime.accelerator.is_local_main_process,
    )

    challenger = PolicyChallenger(
        unet=runtime.unet,
        optimizer=runtime.optimizer,
        validator=runtime.workload.validator,
        noise_scheduler=runtime.noise_scheduler,
        weight_dtype=runtime.weight_dtype,
        learning_rate=runtime.spec.learning_rate,
    )
    state.policy, challenger_log = challenger.choose(
        current_policy=state.policy,
        frozen_batch=FrozenBatch(
            batch=batch,
            noise=current_loss_context.noise,
            timesteps=current_loss_context.timesteps,
        ),
        baseline=validation_report,
    )
    challenger_log["rank_probe"] = maybe_grow_lora_rank(
        unet=runtime.unet,
        optimizer=runtime.optimizer,
        scheduler=runtime.scheduler,
        validation_skeleton=runtime.workload.validator.skeleton,
        baseline_report=validation_report,
        noise_scheduler=runtime.noise_scheduler,
        weight_dtype=runtime.weight_dtype,
        should_probe=not improved,
        base_learning_rate=runtime.spec.learning_rate,
        lora_plus_ratio=state.policy.lora_plus_ratio,
        validation_loss_for_item=runtime.workload.validator.loss,
        validation_loss_tensor_for_item=runtime.workload.validator.loss_tensor,
        free_cuda_bytes=_cuda_free_bytes(),
    )
    write_policy_challenger_report_if_main(
        probe_log_path=runtime.paths.probe_log_path,
        step=state.global_step,
        payload=challenger_log,
        should_log=runtime.accelerator.is_local_main_process,
    )


def _write_final_lora_weights(*, runtime: _TrainingRuntime) -> None:
    runtime.accelerator.wait_for_everyone()
    if runtime.accelerator.is_main_process:
        if runtime.paths.best_weights_path.exists():
            shutil.copy2(runtime.paths.best_weights_path, runtime.paths.final_weights_path)
        else:
            save_named_lora_weights(
                unet=runtime.accelerator.unwrap_model(runtime.unet),
                output_dir=runtime.spec.output_dir,
                filename=LORA_WEIGHTS_NAME,
            )
    runtime.accelerator.end_training()


def _backend_result(*, paths: _BackendPaths) -> BackendResult:
    if not paths.final_weights_path.exists():
        raise LorakitError(
            f"Diffusers backend did not write LoRA weights: {paths.final_weights_path}"
        )
    if paths.probe_log_path.exists():
        write_probe_summary(
            probe_log_path=paths.probe_log_path,
            summary_path=paths.probe_summary_path,
        )
    return BackendResult(
        model_path=paths.final_weights_path,
        artifact_paths=paths.artifact_paths(),
    )


# ---------------------------------------------------------------------------
# Validation and loading
# ---------------------------------------------------------------------------


def _validate_spec(spec: BackendSpec) -> None:
    if spec.steps <= 0:
        raise LorakitError("Training steps must be greater than zero")
    if spec.rank <= 0:
        raise LorakitError("LoRA rank must be greater than zero")
    if spec.batch_size <= 0:
        raise LorakitError("Batch size must be greater than zero")
    if spec.gradient_accumulation <= 0:
        raise LorakitError("Gradient accumulation must be greater than zero")
    if spec.resolution <= 0:
        raise LorakitError("Resolution must be greater than zero")


def _enable_memory_efficient_attention(unet: UNet2DConditionModel) -> str:
    if find_spec("xformers") is None:
        return "torch"
    unet.enable_xformers_memory_efficient_attention()
    return "xformers"


def _cast_trainable_parameters_to_fp32(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)


def _assert_trainable_parameters_are_fp32(module: torch.nn.Module) -> None:
    bad = [
        (name, parameter.dtype)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.dtype != torch.float32
    ]
    if bad:
        preview = ", ".join(f"{name}: {dtype}" for name, dtype in bad[:5])
        raise LorakitError(
            "Trainable parameters must be fp32 under fp16 mixed precision. "
            f"Found: {preview}"
        )


def _accelerate_mixed_precision(value: str) -> str:
    if value == "no":
        return "no"
    if value in {"fp16", "bf16"}:
        return value
    raise LorakitError(f"Unsupported mixed precision: {value}")


def _weight_dtype(accelerator: Accelerator) -> torch.dtype:
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cuda_free_bytes() -> int | None:
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def _cuda_memory_snapshot() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return {
        "free_cuda_bytes": int(free),
        "total_cuda_bytes": int(total),
        "allocated_cuda_bytes": int(torch.cuda.memory_allocated()),
        "reserved_cuda_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_cuda_bytes": int(torch.cuda.max_memory_reserved()),
    }
