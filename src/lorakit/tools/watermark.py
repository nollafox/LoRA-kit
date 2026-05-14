"""Text-like watermark removal for prepared image datasets."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Protocol, Sequence, Tuple

from PIL import Image, ImageFilter, ImageOps

from lorakit.errors import LorakitError


LOGGER = logging.getLogger(__name__)

cv2: Any
easyocr: Any
np: Any
torch: Any

WATERMARK_MODEL_DIR_NAME = "watermark"
EASYOCR_MODEL_DIR_NAME = "easyocr"
INPAINT_RADIUS = 3.0

Point = Tuple[int, int]
Polygon = List[Point]
RoleDistribution = Tuple[float, float]  # (overlay_like, content_like)


class WatermarkRemover(Protocol):
    def process_file(self, input_path: Path, output_path: Path) -> str:
        """Write a cleaned image to output_path.

        Raises:
            LorakitError: If watermark dependencies or model files are missing.
        """


def build_watermark_remover(
    *,
    models_dir: Path,
    cache_dir: Path,
    allow_download: bool,
) -> WatermarkRemover:
    return TextWatermarkCleaner(
        models_dir=models_dir,
        cache_dir=cache_dir,
        allow_download=allow_download,
    )


def ensure_watermark_models(*, models_dir: Path, cache_dir: Path) -> None:
    TextWatermarkCleaner(
        models_dir=models_dir,
        cache_dir=cache_dir,
        allow_download=True,
    )


def _load_dependencies() -> None:
    global cv2
    global easyocr
    global np
    global torch

    try:
        import cv2 as cv2_module
        import easyocr as easyocr_module
        import numpy as numpy_module
        import torch as torch_module
    except ImportError as error:
        raise LorakitError(
            "Watermark removal requires opencv-python, easyocr, numpy, and torch. "
            "Install lorakit with its normal dependencies before running prepare "
            "with --remove-watermarks."
        ) from error

    cv2 = cv2_module
    easyocr = easyocr_module
    np = numpy_module
    torch = torch_module


@dataclass(frozen=True)
class FrameMask:
    name: str
    mask: np.ndarray
    weight: float


@dataclass(frozen=True)
class RoleEvidence:
    name: str
    distribution: RoleDistribution
    weight: float


@dataclass(frozen=True)
class ComponentDecision:
    component_id: int
    bbox: Tuple[int, int, int, int]
    area: int
    overlay_score: float
    content_score: float
    agreement: float
    dispersion_bits: float
    selected: bool
    evidence: Tuple[RoleEvidence, ...]


class TextWatermarkCleaner:
    """Fixed-configuration watermark/signature remover with role consensus."""

    _SCALES = (0.72, 1.00, 1.38)
    _CONTRAST_FRAME_WEIGHT = 0.85
    _STRUCTURAL_ECHO_WEIGHT = 0.75

    _STROKE_DILATION_KERNEL = 3
    _STROKE_DILATION_ITERATIONS = 1
    _SIGNATURE_CLOSE_KERNEL = (9, 3)
    _SIGNATURE_CLOSE_ITERATIONS = 1

    _MIN_COMPONENT_AREA_ABSOLUTE = 10
    _MIN_COMPONENT_AREA_FRACTION = 0.000012
    _MAX_COMPONENT_AREA_FRACTION = 0.22

    _MIN_ROLE_AGREEMENT = 0.60
    _MIN_OVERLAY_SCORE = 0.56
    _MAX_ROLE_DISPERSION_BITS = 1.20
    _STRONG_OVERLAY_SCORE = 0.68

    _MAX_COUNTERFACTUAL_COMPONENTS = 24
    _PROBE_DILATION_KERNEL = 5
    _PROBE_BLUR_RADIUS = 0.50

    _FINAL_DILATION_KERNEL = 5
    _FINAL_DILATION_ITERATIONS = 1
    _FINAL_CLOSE_KERNEL = 5
    _MASK_BLUR_RADIUS = 0.55

    _EPS = 1e-12

    def __init__(
        self,
        *,
        models_dir: Path,
        cache_dir: Path,
        allow_download: bool,
    ) -> None:
        _load_dependencies()
        self.models_dir = models_dir / WATERMARK_MODEL_DIR_NAME
        self.easyocr_model_dir = self.models_dir / EASYOCR_MODEL_DIR_NAME
        self.easyocr_user_network_dir = self.easyocr_model_dir / "user_network"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.easyocr_model_dir.mkdir(parents=True, exist_ok=True)
        self.easyocr_user_network_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "torch").mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(cache_dir / "torch")
        self.gpu = torch.cuda.is_available()
        LOGGER.info("Using %s for EasyOCR detector.", "CUDA" if self.gpu else "CPU")
        self.reader = self._load_easyocr_reader(allow_download=allow_download)

    def _load_easyocr_reader(self, *, allow_download: bool) -> object:
        try:
            return easyocr.Reader(
                ["en"],
                gpu=self.gpu,
                detector=True,
                recognizer=False,
                verbose=False,
                model_storage_directory=str(self.easyocr_model_dir),
                user_network_directory=str(self.easyocr_user_network_dir),
                download_enabled=allow_download,
            )
        except (FileNotFoundError, RuntimeError, OSError, ValueError) as error:
            raise LorakitError(
                "Watermark detector models are not installed. Run "
                "`lorakit install --with-models` from this project before "
                "preparing with watermark removal enabled."
            ) from error

    def process_file(self, input_path: Path, output_path: Path) -> str:
        with Image.open(input_path) as raw:
            raw = ImageOps.exif_transpose(raw)
            original_alpha = raw.getchannel("A").copy() if "A" in raw.getbands() else None
            image = raw.convert("RGB")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_masks = self._build_frame_masks(image)

        if not any(np.any(frame.mask) for frame in frame_masks):
            self._save_image(image=image, alpha=original_alpha, mask=None, output_path=output_path)
            return "copied unchanged: no text-like regions detected"

        selected_mask, decisions = self._decide_components(image, frame_masks, original_alpha)
        if not np.any(selected_mask):
            self._save_image(image=image, alpha=original_alpha, mask=None, output_path=output_path)
            return "copied unchanged: no overlay-like text role reached consensus"

        final_mask = self._finalize_mask(selected_mask)
        restored = self._inpaint_exact(image, final_mask)
        self._save_image(image=restored, alpha=original_alpha, mask=np.asarray(final_mask, dtype=np.uint8), output_path=output_path)

        selected = [decision for decision in decisions if decision.selected]
        mean_dispersion = float(np.mean([decision.dispersion_bits for decision in selected])) if selected else 0.0
        mean_q = float(np.mean([decision.overlay_score for decision in selected])) if selected else 0.0
        return f"restored using {len(selected)} selected region(s); mean overlay_score={mean_q:.3f}; mean dispersion={mean_dispersion:.3f} bits"

    def _build_frame_masks(self, image: Image.Image) -> List[FrameMask]:
        width, height = image.size
        frames: List[FrameMask] = []
        for scale in self._SCALES:
            frames.append(FrameMask(f"rgb@{scale:.2f}", self._detect_mask_at_scale(image, scale), 1.0))

        contrast = self._contrast_frame(image)
        frames.append(FrameMask("contrast@1.00", self._detect_mask_at_scale(contrast, 1.0), self._CONTRAST_FRAME_WEIGHT))

        stacked = np.stack([(frame.mask > 0).astype(np.uint8) for frame in frames], axis=0)
        votes = np.sum(stacked, axis=0)
        echo = np.where(votes >= 2, 255, 0).astype(np.uint8)
        echo = self._dilate(echo, kernel_size=3, iterations=1)
        frames.append(FrameMask("structural_echo", self._ensure_mask_size(echo, width, height), self._STRUCTURAL_ECHO_WEIGHT))
        return [FrameMask(frame.name, self._ensure_mask_size(frame.mask, width, height), frame.weight) for frame in frames]

    def _detect_mask_at_scale(self, image: Image.Image, scale: float) -> np.ndarray:
        width, height = image.size
        if not math.isclose(scale, 1.0):
            scaled_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            working = image.resize(scaled_size, Image.Resampling.LANCZOS)
        else:
            working = image
        polygons = self._detect_text_polygons(working)
        mask = self._polygons_to_binary_mask(polygons, working.size)
        if not math.isclose(scale, 1.0):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask.astype(np.uint8, copy=False)

    def _detect_text_polygons(self, image: Image.Image) -> List[Polygon]:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        horizontal, free = self.reader.detect(rgb)
        width, height = image.size
        polygons: List[Polygon] = []
        for rect in self._flatten_horizontal_boxes(horizontal):
            polygon = self._rectangle_to_polygon(rect, width, height)
            if polygon is not None:
                polygons.append(polygon)
        for box in self._flatten_free_boxes(free):
            polygon = self._normalize_polygon(box, width, height)
            if polygon is not None:
                polygons.append(polygon)
        return self._deduplicate_near_polygons(polygons)

    @classmethod
    def _polygons_to_binary_mask(cls, polygons: Sequence[Polygon], size: Tuple[int, int]) -> np.ndarray:
        width, height = size
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons:
            pts = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
        if np.any(mask):
            mask = cls._dilate(mask, kernel_size=cls._STROKE_DILATION_KERNEL, iterations=cls._STROKE_DILATION_ITERATIONS)
            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, cls._SIGNATURE_CLOSE_KERNEL)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=cls._SIGNATURE_CLOSE_ITERATIONS)
        return mask.astype(np.uint8, copy=False)

    @staticmethod
    def _contrast_frame(image: Image.Image) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        boosted = cv2.merge((l_channel, a_channel, b_channel))
        return Image.fromarray(cv2.cvtColor(boosted, cv2.COLOR_LAB2RGB))

    def _decide_components(self, image: Image.Image, frame_masks: Sequence[FrameMask], alpha: Optional[Image.Image]) -> Tuple[np.ndarray, Tuple[ComponentDecision, ...]]:
        height, width = frame_masks[0].mask.shape
        image_area = width * height
        union = np.zeros((height, width), dtype=np.uint8)
        for frame in frame_masks:
            union = cv2.bitwise_or(union, frame.mask)
        union = self._dilate(union, kernel_size=3, iterations=1)

        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats((union > 0).astype(np.uint8), connectivity=8)
        alpha_np = np.asarray(alpha, dtype=np.uint8) if alpha is not None else None
        image_np = np.asarray(image.convert("RGB"), dtype=np.uint8)
        component_ids = self._rank_component_ids_for_probe(labels_count, stats, image_area, width, height)
        selected_mask = np.zeros_like(union, dtype=np.uint8)
        decisions: List[ComponentDecision] = []

        for component_id in component_ids:
            x, y, w, h, area = [int(v) for v in stats[component_id]]
            if not self._component_area_allowed(area, image_area):
                continue
            component = labels == component_id
            bbox = (x, y, w, h)
            probe_mask = self._component_probe_mask(component)
            counterfactual = self._counterfactual_patch(image, probe_mask, bbox)
            evidence = self._role_evidence(image_np=image_np, component=component, bbox=bbox, frame_masks=frame_masks, alpha_np=alpha_np, counterfactual=counterfactual)
            overlay_score, content_score, agreement, dispersion_bits = self._role_consensus(evidence)
            selected = self._accept_role_consensus(overlay_score=overlay_score, agreement=agreement, dispersion_bits=dispersion_bits, evidence=evidence)
            if selected:
                selected_mask[component] = 255
            decisions.append(ComponentDecision(component_id, bbox, area, overlay_score, content_score, agreement, dispersion_bits, selected, evidence))
        return selected_mask, tuple(decisions)

    def _rank_component_ids_for_probe(self, labels_count: int, stats: np.ndarray, image_area: int, width: int, height: int) -> List[int]:
        rows: List[Tuple[float, int]] = []
        for component_id in range(1, labels_count):
            x, y, w, h, area = [int(v) for v in stats[component_id]]
            if not self._component_area_allowed(area, image_area):
                continue
            edge = self._edge_affinity((x, y, w, h), width, height)
            score = area * (1.0 + 0.25 * edge)
            rows.append((-score, component_id))
        rows.sort()
        return [component_id for _, component_id in rows[: self._MAX_COUNTERFACTUAL_COMPONENTS]]

    def _role_evidence(self, *, image_np: np.ndarray, component: np.ndarray, bbox: Tuple[int, int, int, int], frame_masks: Sequence[FrameMask], alpha_np: Optional[np.ndarray], counterfactual: Optional[np.ndarray]) -> Tuple[RoleEvidence, ...]:
        height, width = component.shape
        image_area = width * height
        area = int(np.count_nonzero(component))
        detector_support = self._detector_support(component, frame_masks)
        placement_ext = self._placement_overlay_probability(bbox, width, height)
        scale_ext = self._scale_overlay_probability(area, bbox, image_area, width, height)
        alpha_ext = self._alpha_overlay_probability(component, alpha_np)
        layer_ext = self._layer_simplicity_probability(image_np, component, bbox)
        salience_ext = self._salience_overlay_probability(image_np, component, bbox)
        repetition_ext = self._repetition_overlay_probability(component, frame_masks)
        counterfactual_ext = self._counterfactual_overlay_probability(image_np, component, bbox, counterfactual)
        return (
            self._evidence("detector_stability", detector_support, 1.20),
            self._evidence("placement_metadata", placement_ext, 0.95),
            self._evidence("scale_subordination", scale_ext, 1.10),
            self._evidence("alpha_layer", alpha_ext, 0.80),
            self._evidence("overlay_simplicity", layer_ext, 1.05),
            self._evidence("low_scene_salience", salience_ext, 1.05),
            self._evidence("repetition_or_mark_family", repetition_ext, 0.70),
            self._evidence("counterfactual_boundary", counterfactual_ext, 1.25),
        )

    @classmethod
    def _evidence(cls, name: str, overlay_probability: float, weight: float) -> RoleEvidence:
        p = float(np.clip(overlay_probability, 0.01, 0.99))
        return RoleEvidence(name, (p, 1.0 - p), float(weight))

    @classmethod
    def _role_consensus(cls, evidence: Sequence[RoleEvidence]) -> Tuple[float, float, float, float]:
        """
        Weighted square-root consensus over two roles.

        Each evidence channel votes for an overlay-like or content-like role.
        The consensus distribution is the normalized weighted square-root
        barycenter of those votes. The returned agreement measures how far the
        most skeptical channel is from the consensus, which keeps noisy evidence
        visible without letting one conservative channel veto strong cases.
        """
        if not evidence:
            return 0.0, 1.0, 0.0, math.inf

        ps = np.asarray([item.distribution[0] for item in evidence], dtype=np.float64)
        weights = np.asarray([item.weight for item in evidence], dtype=np.float64)
        weights = weights / max(cls._EPS, float(weights.sum()))

        root_ext = float(np.sum(weights * np.sqrt(np.clip(ps, cls._EPS, 1.0))))
        root_int = float(np.sum(weights * np.sqrt(np.clip(1.0 - ps, cls._EPS, 1.0))))
        overlay_score_raw = root_ext * root_ext
        content_score_raw = root_int * root_int
        denom = max(cls._EPS, overlay_score_raw + content_score_raw)
        overlay_score = float(np.clip(overlay_score_raw / denom, cls._EPS, 1.0 - cls._EPS))

        local = np.sqrt(ps * overlay_score) + np.sqrt((1.0 - ps) * (1.0 - overlay_score))
        alpha = float(np.clip(np.min(local), cls._EPS, 1.0))
        return overlay_score, 1.0 - overlay_score, alpha, float(-math.log2(alpha))

    def _accept_role_consensus(self, *, overlay_score: float, agreement: float, dispersion_bits: float, evidence: Sequence[RoleEvidence]) -> bool:
        evidence_map = {item.name: item.distribution[0] for item in evidence}
        detector = evidence_map.get("detector_stability", 0.5)
        counterfactual = evidence_map.get("counterfactual_boundary", 0.5)
        scale = evidence_map.get("scale_subordination", 0.5)
        placement = evidence_map.get("placement_metadata", 0.5)
        alpha = evidence_map.get("alpha_layer", 0.5)
        layer = evidence_map.get("overlay_simplicity", 0.5)
        salience = evidence_map.get("low_scene_salience", 0.5)
        repetition = evidence_map.get("repetition_or_mark_family", 0.5)

        # True content text usually looks big, central, opaque, salient, and
        # non-repeated. This is a veto for exactly that role, not a general
        # reluctance to remove text.
        probable_content_text = (
            detector >= 0.54
            and scale <= 0.34
            and placement <= 0.42
            and alpha <= 0.56
            and repetition <= 0.58
            and salience <= 0.42
            and overlay_score < 0.74
        )
        if probable_content_text:
            return False

        # Agreement remains a quality signal, but no longer a hard global veto
        # for high-confidence overlay cases.
        if overlay_score >= self._STRONG_OVERLAY_SCORE and detector >= 0.50:
            return True

        # Main removal path: detector says text, the role posterior leans
        # overlay-like, and at least one independent channel explains it as an
        # overlay/metadata object.
        independent_support = max(counterfactual, placement, scale, alpha, layer, repetition)
        if overlay_score >= self._MIN_OVERLAY_SCORE and detector >= 0.52 and independent_support >= 0.55:
            return True

        # Tiny footer/corner artist marks often have weak counterfactual gains
        # because the probe is small, but their placement+scale code length is
        # strongly metadata-like.
        if detector >= 0.48 and placement >= 0.66 and scale >= 0.58 and layer >= 0.50:
            return True

        # Large translucent or repeated watermarks are allowed through even if
        # salience/content channels complain.
        if detector >= 0.50 and (alpha >= 0.66 or repetition >= 0.68) and overlay_score >= 0.52:
            return True

        # Last-resort recall path: this approximates the v3 behavior but keeps
        # the foreground-content veto above.
        if detector >= 0.66 and layer >= 0.60 and (placement >= 0.50 or scale >= 0.50 or counterfactual >= 0.50):
            return True

        return False

    def _detector_support(self, component: np.ndarray, frame_masks: Sequence[FrameMask]) -> float:
        values = np.asarray([self._occupancy(component, frame.mask) for frame in frame_masks], dtype=np.float64)
        weights = np.asarray([frame.weight for frame in frame_masks], dtype=np.float64)
        support = float((values @ weights) / max(self._EPS, float(weights.sum())))
        agreement, _ = self._finite_bernoulli_agreement(values)
        return float(np.clip(0.25 + 0.60 * support + 0.15 * agreement, 0.01, 0.99))

    @staticmethod
    def _placement_overlay_probability(bbox: Tuple[int, int, int, int], width: int, height: int) -> float:
        x, y, w, h = bbox
        cx = (x + 0.5 * w) / max(1.0, width)
        cy = (y + 0.5 * h) / max(1.0, height)
        margin_distance = min(x / max(1.0, width), y / max(1.0, height), (width - x - w) / max(1.0, width), (height - y - h) / max(1.0, height))
        edge_affinity = float(np.clip(1.0 - margin_distance / 0.18, 0.0, 1.0))
        footer = float(np.clip((cy - 0.74) / 0.20, 0.0, 1.0))
        header = float(np.clip((0.20 - cy) / 0.20, 0.0, 1.0))
        corner = edge_affinity * float(max(abs(cx - 0.5), abs(cy - 0.5)) * 2.0)
        center_penalty = float(np.clip(1.0 - math.hypot(cx - 0.5, cy - 0.5) / 0.50, 0.0, 1.0))
        p = 0.18 + 0.35 * edge_affinity + 0.26 * max(footer, header) + 0.17 * corner - 0.18 * center_penalty
        return float(np.clip(p, 0.01, 0.99))

    @staticmethod
    def _scale_overlay_probability(area: int, bbox: Tuple[int, int, int, int], image_area: int, width: int, height: int) -> float:
        _, _, w, h = bbox
        area_frac = area / max(1.0, image_area)
        height_frac = h / max(1.0, height)
        width_frac = w / max(1.0, width)
        slenderness = max(w, h) / max(1.0, min(w, h))
        small = float(np.clip((0.040 - area_frac) / 0.040, 0.0, 1.0))
        low_height = float(np.clip((0.18 - height_frac) / 0.18, 0.0, 1.0))
        low_width = float(np.clip((0.55 - width_frac) / 0.55, 0.0, 1.0))
        signature_shape = float(np.clip((slenderness - 2.4) / 4.0, 0.0, 1.0))
        huge_penalty = float(np.clip((max(height_frac, width_frac) - 0.32) / 0.34, 0.0, 1.0))
        p = 0.10 + 0.38 * small + 0.22 * low_height + 0.12 * low_width + 0.20 * signature_shape - 0.34 * huge_penalty
        return float(np.clip(p, 0.01, 0.99))

    @staticmethod
    def _alpha_overlay_probability(component: np.ndarray, alpha_np: Optional[np.ndarray]) -> float:
        if alpha_np is None:
            return 0.50
        values = alpha_np[component]
        if values.size == 0:
            return 0.50
        semi = np.logical_and(values > 8, values < 248)
        semi_fraction = float(np.count_nonzero(semi) / values.size)
        return float(np.clip(0.42 + 0.50 * semi_fraction, 0.01, 0.99))

    @staticmethod
    def _layer_simplicity_probability(image_np: np.ndarray, component: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        x, y, w, h = bbox
        patch = image_np[y : y + h, x : x + w]
        mask = component[y : y + h, x : x + w]
        pixels = patch[mask]
        if pixels.size == 0:
            return 0.50
        quantized = (pixels // 32).astype(np.uint8)
        packed = quantized[:, 0].astype(np.int32) * 64 + quantized[:, 1].astype(np.int32) * 8 + quantized[:, 2].astype(np.int32)
        counts = np.bincount(packed, minlength=512).astype(np.float64)
        probs = counts[counts > 0] / max(1.0, float(counts.sum()))
        entropy = -float(np.sum(probs * np.log2(probs))) if probs.size else 0.0
        entropy_score = float(np.clip((5.2 - entropy) / 5.2, 0.0, 1.0))
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 70, 160)
        edge_density = float(np.count_nonzero(edges[mask]) / max(1, np.count_nonzero(mask)))
        stroke_score = float(np.clip(edge_density / 0.22, 0.0, 1.0))
        p = 0.22 + 0.48 * entropy_score + 0.20 * stroke_score
        return float(np.clip(p, 0.01, 0.99))

    @staticmethod
    def _salience_overlay_probability(image_np: np.ndarray, component: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        x, y, w, h = bbox
        patch = image_np[y : y + h, x : x + w]
        mask = component[y : y + h, x : x + w]
        if not np.any(mask):
            return 0.50
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
        inside = gray[mask]
        dilated = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), dtype=np.uint8), iterations=1).astype(bool)
        ring = dilated & ~mask
        outside = gray[ring] if np.any(ring) else gray[~mask]
        if outside.size == 0:
            return 0.50
        contrast = abs(float(np.mean(inside)) - float(np.mean(outside))) / 255.0
        var_ratio = float(np.std(inside) / max(1.0, np.std(outside)))
        high_salience = float(np.clip(0.70 * contrast + 0.20 * min(var_ratio, 2.0) / 2.0, 0.0, 1.0))
        return float(np.clip(0.74 - 0.56 * high_salience, 0.01, 0.99))

    @staticmethod
    def _repetition_overlay_probability(component: np.ndarray, frame_masks: Sequence[FrameMask]) -> float:
        if not frame_masks:
            return 0.50
        union = np.zeros_like(frame_masks[0].mask, dtype=np.uint8)
        for frame in frame_masks:
            union = cv2.bitwise_or(union, frame.mask)
        labels_count, _, stats, _ = cv2.connectedComponentsWithStats((union > 0).astype(np.uint8), connectivity=8)
        area = max(1, int(np.count_nonzero(component)))
        similar = 0
        for component_id in range(1, labels_count):
            other_area = int(stats[component_id, cv2.CC_STAT_AREA])
            ratio = other_area / area
            if 0.35 <= ratio <= 2.85:
                similar += 1
        return float(np.clip(0.40 + 0.11 * min(similar - 1, 5), 0.01, 0.99))

    def _counterfactual_overlay_probability(self, image_np: np.ndarray, component: np.ndarray, bbox: Tuple[int, int, int, int], counterfactual: Optional[np.ndarray]) -> float:
        if counterfactual is None:
            return 0.50
        x, y, w, h = bbox
        probe = self._component_probe_mask(component)[y : y + h, x : x + w] > 0
        original_patch = image_np[y : y + h, x : x + w].astype(np.float32)
        restored_patch = counterfactual.astype(np.float32)
        if original_patch.shape != restored_patch.shape or not np.any(probe):
            return 0.50
        before_boundary = self._boundary_discontinuity(original_patch, probe)
        after_boundary = self._boundary_discontinuity(restored_patch, probe)
        boundary_gain = before_boundary - after_boundary
        before_texture = self._texture_surprise(original_patch, probe)
        after_texture = self._texture_surprise(restored_patch, probe)
        texture_gain = before_texture - after_texture
        p = 0.50 + 0.33 * math.tanh(2.8 * boundary_gain) + 0.22 * math.tanh(2.2 * texture_gain)
        return float(np.clip(p, 0.01, 0.99))

    @staticmethod
    def _boundary_discontinuity(patch: np.ndarray, mask: np.ndarray) -> float:
        gray = cv2.cvtColor(np.clip(patch, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        kernel = np.ones((3, 3), dtype=np.uint8)
        outer = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool) & ~mask
        inner = mask & ~cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        if not np.any(inner) or not np.any(outer):
            return 0.0
        return abs(float(np.mean(gray[inner])) - float(np.mean(gray[outer])))

    @staticmethod
    def _texture_surprise(patch: np.ndarray, mask: np.ndarray) -> float:
        gray = cv2.cvtColor(np.clip(patch, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        kernel = np.ones((7, 7), dtype=np.uint8)
        ring = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool) & ~mask
        if not np.any(mask) or not np.any(ring):
            return 0.0
        mean_gap = abs(float(np.mean(gray[mask])) - float(np.mean(gray[ring])))
        std_gap = abs(float(np.std(gray[mask])) - float(np.std(gray[ring])))
        return 0.70 * mean_gap + 0.30 * std_gap

    def _counterfactual_patch(self, image: Image.Image, probe_mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x, y, w, h = bbox
        pad = max(24, int(round(0.28 * max(w, h))))
        width, height = image.size
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(width, x + w + pad), min(height, y + h + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image.crop((x1, y1, x2, y2)).convert("RGB")
        crop_mask_np = probe_mask[y1:y2, x1:x2].astype(np.uint8)
        if not np.any(crop_mask_np):
            return None
        crop_mask = Image.fromarray(crop_mask_np, mode="L").filter(ImageFilter.GaussianBlur(self._PROBE_BLUR_RADIUS))
        try:
            restored_crop = self._inpaint_exact(crop, crop_mask)
        except (RuntimeError, ValueError, OSError):
            return None
        local_x, local_y = x - x1, y - y1
        patch = restored_crop.crop((local_x, local_y, local_x + w, local_y + h))
        return np.asarray(patch.convert("RGB"), dtype=np.uint8)

    @classmethod
    def _component_probe_mask(cls, component: np.ndarray) -> np.ndarray:
        mask = np.where(component, 255, 0).astype(np.uint8)
        return cls._dilate(mask, kernel_size=cls._PROBE_DILATION_KERNEL, iterations=1)

    @classmethod
    def _finalize_mask(cls, selected: np.ndarray) -> Image.Image:
        mask = selected.astype(np.uint8, copy=False)
        if np.any(mask):
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cls._FINAL_CLOSE_KERNEL, cls._FINAL_CLOSE_KERNEL))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
            mask = cls._dilate(mask, kernel_size=cls._FINAL_DILATION_KERNEL, iterations=cls._FINAL_DILATION_ITERATIONS)
        return Image.fromarray(mask, mode="L").filter(ImageFilter.GaussianBlur(cls._MASK_BLUR_RADIUS))

    def _inpaint_exact(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        mask = mask.convert("L")
        if image.size != mask.size:
            raise ValueError(f"Image/mask mismatch: image={image.size}, mask={mask.size}")
        image_np = np.asarray(image, dtype=np.uint8)
        mask_np = np.asarray(mask, dtype=np.uint8)
        binary_mask = np.where(mask_np > 8, 255, 0).astype(np.uint8)
        if not np.any(binary_mask):
            return image
        bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        restored = cv2.inpaint(bgr, binary_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))

    @classmethod
    def _finite_bernoulli_agreement(cls, probabilities: Sequence[float]) -> Tuple[float, float]:
        ps = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
        if ps.size == 0:
            return 0.0, math.inf
        candidates = np.unique(np.r_[np.linspace(0.0, 1.0, 1001), ps, np.mean(ps), np.median(ps)])
        best = 0.0
        for q in candidates:
            bc = np.sqrt(ps * q) + np.sqrt((1.0 - ps) * (1.0 - q))
            worst = float(np.min(bc))
            if worst > best:
                best = worst
        best = float(np.clip(best, cls._EPS, 1.0))
        return best, float(-math.log2(best))

    @staticmethod
    def _occupancy(component: np.ndarray, mask: np.ndarray) -> float:
        denom = int(np.count_nonzero(component))
        if denom <= 0:
            return 0.0
        return float(np.count_nonzero(component & (mask > 0)) / denom)

    @classmethod
    def _component_area_allowed(cls, area: int, image_area: int) -> bool:
        min_area = max(cls._MIN_COMPONENT_AREA_ABSOLUTE, int(round(cls._MIN_COMPONENT_AREA_FRACTION * image_area)))
        max_area = max(min_area, int(round(cls._MAX_COMPONENT_AREA_FRACTION * image_area)))
        return min_area <= int(area) <= max_area

    @staticmethod
    def _edge_affinity(bbox: Tuple[int, int, int, int], width: int, height: int) -> float:
        x, y, w, h = bbox
        x2, y2 = x + w, y + h
        distance = min(x, y, max(0, width - x2), max(0, height - y2))
        scale = max(1.0, 0.18 * min(width, height))
        return float(np.clip(1.0 - distance / scale, 0.0, 1.0))

    @staticmethod
    def _ensure_mask_size(mask: np.ndarray, width: int, height: int) -> np.ndarray:
        if mask.shape == (height, width):
            return mask.astype(np.uint8, copy=False)
        return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _dilate(mask: np.ndarray, *, kernel_size: int, iterations: int) -> np.ndarray:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)

    @staticmethod
    def _flatten_horizontal_boxes(raw) -> Iterable[Sequence[float]]:
        if raw is None:
            return []
        boxes: List[Sequence[float]] = []
        for item in raw:
            arr = np.asarray(item, dtype=object)
            if arr.ndim == 1 and len(item) == 4 and all(np.isscalar(x) for x in item):
                boxes.append(item)
            else:
                for sub in item:
                    if len(sub) == 4:
                        boxes.append(sub)
        return boxes

    @staticmethod
    def _flatten_free_boxes(raw) -> Iterable[Sequence[Sequence[float]]]:
        if raw is None:
            return []
        boxes: List[Sequence[Sequence[float]]] = []
        for item in raw:
            arr = np.asarray(item, dtype=object)
            if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] == 2:
                boxes.append(item)
            else:
                for sub in item:
                    arr_sub = np.asarray(sub, dtype=object)
                    if arr_sub.ndim == 2 and arr_sub.shape[0] >= 3 and arr_sub.shape[1] == 2:
                        boxes.append(sub)
        return boxes

    @classmethod
    def _rectangle_to_polygon(cls, raw_box: Sequence[float], width: int, height: int) -> Optional[Polygon]:
        if len(raw_box) != 4:
            return None
        x_min, x_max, y_min, y_max = [float(v) for v in raw_box]
        polygon = [(int(round(x_min)), int(round(y_min))), (int(round(x_max)), int(round(y_min))), (int(round(x_max)), int(round(y_max))), (int(round(x_min)), int(round(y_max)))]
        return cls._normalize_polygon(polygon, width, height)

    @staticmethod
    def _normalize_polygon(raw_box, width: int, height: int) -> Optional[Polygon]:
        arr = np.asarray(raw_box, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
            return None
        arr[:, 0] = np.clip(arr[:, 0], 0, max(0, width - 1))
        arr[:, 1] = np.clip(arr[:, 1], 0, max(0, height - 1))
        polygon = [(int(round(x)), int(round(y))) for x, y in arr]
        area = abs(cv2.contourArea(np.asarray(polygon, dtype=np.float32)))
        if area < 9.0:
            return None
        return polygon

    @classmethod
    def _deduplicate_near_polygons(cls, polygons: Sequence[Polygon]) -> List[Polygon]:
        kept: List[Polygon] = []
        kept_boxes: List[Tuple[int, int, int, int]] = []
        for polygon in polygons:
            box = cls._bbox_from_polygon(polygon)
            if any(cls._box_iou(box, old) > 0.86 for old in kept_boxes):
                continue
            kept.append(polygon)
            kept_boxes.append(box)
        return kept

    @staticmethod
    def _bbox_from_polygon(polygon: Polygon) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
        area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
        denom = area_a + area_b - inter
        return float(inter / denom) if denom > 0 else 0.0

    @staticmethod
    def _save_image(*, image: Image.Image, alpha: Optional[Image.Image], mask: Optional[np.ndarray], output_path: Path) -> None:
        ext = output_path.suffix.lower()
        out = image.convert("RGB")
        if alpha is not None and ext in {".png", ".webp", ".tif", ".tiff"}:
            rgba = out.convert("RGBA")
            if alpha.size != rgba.size:
                alpha = alpha.resize(rgba.size, Image.Resampling.LANCZOS)
            alpha_np = np.asarray(alpha, dtype=np.uint8).copy()
            if mask is not None and np.any(mask):
                alpha_np[mask > 8] = 255
            rgba.putalpha(Image.fromarray(alpha_np, mode="L"))
            rgba.save(output_path)
            return
        if ext in {".jpg", ".jpeg"}:
            out.save(output_path, quality=95, subsampling=0, optimize=True)
        elif ext == ".webp":
            out.save(output_path, quality=95, method=6)
        else:
            out.save(output_path)
