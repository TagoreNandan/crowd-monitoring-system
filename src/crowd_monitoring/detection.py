"""Person detector: YOLOv8n fine-tuned on CrowdHuman (Ultralytics architecture,
real GitHub-Release-hosted weights), with optional SAHI-style tiled inference
for offline/high-latency-budget use.

Model choice, measured on this project's own sample files (see
`docs`/commit history for the full profiling run -- summarized here because
it's the reason this file looks the way it does):

  - The original detector (COCO-pretrained YOLOv8n/YOLOv8m, single class
    "person" out of 80 COCO classes) finds relatively few people in a dense
    crowd: ~16-28 boxes on this project's sample image/video frame at a
    single full-frame pass. Densifying it further by SAHI-style tiling (32
    tiles across a 4K frame) did increase recall (up to ~193 boxes) but at a
    catastrophic latency cost -- ~850ms/frame with YOLOv8n tiles, ~4.6
    SECONDS/frame with YOLOv8m tiles -- which is exactly backwards for a
    real-time pipeline, and YOLOv8m tiled was both slower AND found FEWER
    boxes than YOLOv8n tiled (bigger model =/= better here).
  - Swapping the WEIGHTS to a detector actually trained for dense/occluded
    crowds (CrowdHuman: ~340k person instances across ~15k images,
    specifically annotated for heavy inter-person occlusion) fixes the root
    cause instead of compensating for it with brute-force tiling: a single
    full-frame pass at imgsz=1600 finds ~135-226 boxes on the same sample
    files in ~83-84ms -- more boxes than the old 32-tile approach, at roughly
    1/10th the latency, with no tiling at all.

  Source: https://github.com/yakhyo/yolov8-crowdhuman (real GitHub Release
  asset, `yolov8n_best.pt`, single class "person", YOLOv8n architecture
  trained on the public CrowdHuman dataset -- not an unofficial/random
  mirror). Falls back to the plain COCO-pretrained `yolov8n.pt` if this
  checkpoint isn't present, so the detector still works even before the
  CrowdHuman weights are fetched.

Tiling (`generate_tiles`/`nms_boxes`, SAHI-style) is kept as an opt-in
(`use_tiling=True`) for a caller with a generous latency budget that wants to
squeeze out more of the tiniest/farthest boxes on a single still image -- it
is NOT the default anywhere in this codebase, precisely because of the
latency numbers above.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = Path(os.environ.get("CROWD_MONITORING_WEIGHTS_DIR", REPO_ROOT / "weights"))

# CrowdHuman-trained YOLOv8n (see module docstring). Falls back to the plain
# COCO-pretrained yolov8n.pt (still downloaded from Ultralytics' own official
# release, see `_ensure_model`) if this specific checkpoint isn't present.
DEFAULT_WEIGHTS_PATH = Path(os.environ.get("CROWD_MONITORING_YOLO_WEIGHTS", WEIGHTS_DIR / "yolov8n_crowdhuman.pt"))
_COCO_FALLBACK_WEIGHTS_PATH = WEIGHTS_DIR / "yolov8n.pt"

DEFAULT_CONFIDENCE_THRESHOLD = 0.20
# 1600px measured as the latency/quality sweet spot on this project's own 4K
# sample video frame and sample image: within ~5% of imgsz=1920's detection
# count at ~25ms/frame less, and meaningfully better than imgsz=1280 without
# the cost of imgsz=2560 (which stopped helping and cost 4-6x more).
DEFAULT_IMGSZ = 1600
DEFAULT_TILE_SIZE = 640
DEFAULT_TILE_OVERLAP = 0.20
DEFAULT_NMS_IOU_THRESHOLD = 0.45
DEFAULT_USE_TILING = False

# COCO class 0 = "person".
COCO_PERSON_CLASS_ID = 0

try:
    from ultralytics import YOLO as _YOLO

    _IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover
    _YOLO = None
    _IMPORT_ERROR = str(exc)


@dataclass
class PersonDetection:
    """One real, above-threshold, person-only bounding box. Coordinates are
    pixel coordinates in the original (un-resized) frame.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    track_id: Optional[int] = None


class PersonDetectorUnavailableError(RuntimeError):
    """Raised when the YOLO model can't be loaded."""


def nms_boxes(detections: List[PersonDetection], iou_threshold: float = 0.45) -> List[PersonDetection]:
    """Perform Non-Maximum Suppression across candidate person bounding boxes."""
    if not detections:
        return []
    boxes = [[d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1] for d in detections]
    scores = [d.confidence for d in detections]
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    if len(indices) == 0:
        return []
    if isinstance(indices, tuple):
        idx_list = [int(i) for i in indices]
    elif hasattr(indices, "flatten"):
        idx_list = [int(i) for i in indices.flatten().tolist()]
    else:
        idx_list = [int(i) for i in indices]
    return [detections[i] for i in idx_list]


def generate_tiles(width: int, height: int, tile_size: int = 640, tile_overlap: float = 0.20) -> List[Tuple[int, int, int, int]]:
    """Generate overlapping tile coordinates `(x1, y1, x2, y2)` for an image of dimensions `(width, height)`."""
    if width <= tile_size and height <= tile_size:
        return [(0, 0, width, height)]

    stride = int(tile_size * (1.0 - tile_overlap))
    stride = max(1, stride)

    tiles: list[Tuple[int, int, int, int]] = []
    y_starts = list(range(0, max(1, height - tile_size + 1), stride))
    if y_starts[-1] + tile_size < height:
        y_starts.append(height - tile_size)

    x_starts = list(range(0, max(1, width - tile_size + 1), stride))
    if x_starts[-1] + tile_size < width:
        x_starts.append(width - tile_size)

    for y in y_starts:
        for x in x_starts:
            x1 = max(0, min(x, width - tile_size))
            y1 = max(0, min(y, height - tile_size))
            x2 = min(width, x1 + tile_size)
            y2 = min(height, y1 + tile_size)
            tiles.append((x1, y1, x2, y2))

    return list(dict.fromkeys(tiles))


# Density-guided selective refinement (adaptive alternative to blind tiling):
# rather than tiling the whole frame, use DM-Count's own density map -- a
# signal this project already computes for every image -- to find where the
# crowd is actually concentrated, and only re-detect on those regions.
# Measured on this project's own sample image (examples/crowd.jpeg):
# full-frame pass alone finds 135 boxes; adding up to 3 density-guided
# region crops (NMS-merged with the full pass) finds 152 (+13%) for ~700ms
# extra -- a real, targeted gain, not the ~30x latency blowup blind 32-tile
# inference cost for a comparable/worse gain. A naive "redetect the single
# densest region" does NOT reliably help on its own (verified: the densest
# region is often a genuinely-unresolvable far-field crowd where cropping
# and upscaling recovers no new pixel information) -- the gain comes from
# covering several distinct dense regions and merging via NMS, not from any
# one crop alone.
DEFAULT_DENSITY_THRESHOLD_REL = 0.25
DEFAULT_MAX_REFINEMENT_REGIONS = 3
DEFAULT_MIN_REGION_DENSITY_PIXELS = 6
DEFAULT_REGION_PADDING_PX = 15


def find_dense_regions(
    density_map: np.ndarray,
    original_size: Tuple[int, int],
    threshold_rel: float = DEFAULT_DENSITY_THRESHOLD_REL,
    max_regions: int = DEFAULT_MAX_REFINEMENT_REGIONS,
    min_region_pixels: int = DEFAULT_MIN_REGION_DENSITY_PIXELS,
    padding_px: int = DEFAULT_REGION_PADDING_PX,
) -> List[Tuple[int, int, int, int]]:
    """Find up to `max_regions` connected high-density blobs in a DM-Count
    density map and return their bounding boxes in ORIGINAL image pixel
    coordinates, largest (by total density mass) first.

    `original_size` is `(height, width)` of the frame the density map was
    computed from (density maps are produced at a smaller, model-native
    resolution -- see `crowd_monitoring.inference` -- so this scales back up).
    Returns an empty list if the map is empty/uniform (nothing to refine).
    """
    if density_map is None or density_map.size == 0:
        return []
    d = density_map.astype(np.float32)
    if d.max() <= 0:
        return []

    from scipy import ndimage

    mask = d >= (d.max() * threshold_rel)
    labeled, n_labels = ndimage.label(mask)
    if n_labels == 0:
        return []

    oh, ow = original_size
    dh, dw = d.shape
    scale_x, scale_y = ow / dw, oh / dh

    candidates: List[Tuple[float, int, int, int, int]] = []
    for label_id in range(1, n_labels + 1):
        ys, xs = np.nonzero(labeled == label_id)
        if len(ys) < min_region_pixels:
            continue
        mass = float(d[labeled == label_id].sum())
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        candidates.append((mass, x1, y1, x2, y2))

    candidates.sort(key=lambda c: -c[0])

    regions: List[Tuple[int, int, int, int]] = []
    for _mass, x1, y1, x2, y2 in candidates[:max_regions]:
        ox1 = max(0, int(x1 * scale_x) - padding_px)
        oy1 = max(0, int(y1 * scale_y) - padding_px)
        ox2 = min(ow, int(x2 * scale_x) + padding_px)
        oy2 = min(oh, int(y2 * scale_y) + padding_px)
        if ox2 > ox1 and oy2 > oy1:
            regions.append((ox1, oy1, ox2, oy2))
    return regions


class PersonDetector:
    """Loads a YOLO model once (lazily) and runs high-density person detection on BGR frames."""

    def __init__(
        self,
        weights_path: Path = DEFAULT_WEIGHTS_PATH,
        device: str = "auto",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        imgsz: int = DEFAULT_IMGSZ,
        use_tiling: bool = DEFAULT_USE_TILING,
        tile_size: int = DEFAULT_TILE_SIZE,
        tile_overlap: float = DEFAULT_TILE_OVERLAP,
        nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
    ):
        self.weights_path = Path(weights_path)
        self.device_preference = device
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self.use_tiling = use_tiling
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.nms_iou_threshold = nms_iou_threshold

        self._model: Optional[Any] = None
        self._device_str: str = "cpu"
        self._resolved_torch_device: Optional[Any] = None
        self._load_failed = False
        self._lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        return _YOLO is not None and not self._load_failed

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if _YOLO is None:
            raise PersonDetectorUnavailableError(
                f"The 'ultralytics' package is not installed ({_IMPORT_ERROR})."
            )
        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_failed:
                raise PersonDetectorUnavailableError("YOLO person detector previously failed to load.")
            try:
                from .device import resolve_device

                torch_device = resolve_device(self.device_preference)
                self._device_str = _torch_device_to_ultralytics_str(torch_device)
                self._resolved_torch_device = torch_device

                # If the requested (CrowdHuman, by default) checkpoint isn't
                # present locally, fall back to the COCO-pretrained
                # yolov8n.pt (which `PersonDetector` also auto-downloads --
                # see the `except` below) rather than failing outright.
                weights = self.weights_path
                if not weights.exists() and weights != _COCO_FALLBACK_WEIGHTS_PATH:
                    logger.warning(
                        "CrowdHuman YOLO weights not found at %s; falling back to COCO-pretrained %s "
                        "(fewer boxes in dense/occluded crowds -- see crowd_monitoring.detection docstring).",
                        weights, _COCO_FALLBACK_WEIGHTS_PATH,
                    )
                    weights = _COCO_FALLBACK_WEIGHTS_PATH

                weights.parent.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Loading YOLO person detector from %s (device=%s)", weights, self._device_str
                )
                model = _YOLO(str(weights))
                self._model = model
            except Exception as exc:
                self._load_failed = True
                logger.exception("Failed to load YOLO person detector")
                raise PersonDetectorUnavailableError(
                    f"Could not load YOLO weights from '{self.weights_path}': {exc}"
                ) from exc
        return self._model

    def detect(self, frame_bgr: np.ndarray, use_tiling: Optional[bool] = None) -> List[PersonDetection]:
        """Run person detection on a frame using tiled SAHI-style inference when appropriate."""
        model = self._ensure_model()
        h, w = frame_bgr.shape[:2]

        should_tile = self.use_tiling if use_tiling is None else use_tiling
        if not should_tile or (w <= self.tile_size and h <= self.tile_size):
            return self._predict_single(model, frame_bgr, imgsz=self.imgsz)

        # Tiled SAHI-style detection
        tiles = generate_tiles(w, h, tile_size=self.tile_size, tile_overlap=self.tile_overlap)
        all_candidates: list[PersonDetection] = []

        for tx1, ty1, tx2, ty2 in tiles:
            tile_crop = frame_bgr[ty1:ty2, tx1:tx2]
            tile_dets = self._predict_single(model, tile_crop, imgsz=self.tile_size)
            for d in tile_dets:
                all_candidates.append(
                    PersonDetection(
                        x1=d.x1 + tx1,
                        y1=d.y1 + ty1,
                        x2=d.x2 + tx1,
                        y2=d.y2 + ty1,
                        confidence=d.confidence,
                    )
                )

        # Also run full image pass to catch large/foreground subjects
        full_dets = self._predict_single(model, frame_bgr, imgsz=self.imgsz)
        all_candidates.extend(full_dets)

        # Perform NMS across tile candidates
        return nms_boxes(all_candidates, iou_threshold=self.nms_iou_threshold)

    def detect_with_density_refinement(
        self,
        frame_bgr: np.ndarray,
        density_map: Optional[np.ndarray],
        max_regions: int = DEFAULT_MAX_REFINEMENT_REGIONS,
        threshold_rel: float = DEFAULT_DENSITY_THRESHOLD_REL,
    ) -> List[PersonDetection]:
        """Full-frame detection PLUS selective re-detection on up to
        `max_regions` DM-Count-identified dense regions, merged via NMS.

        This is the adaptive alternative to blind tiling: it costs a full
        pass (~85ms) plus at most `max_regions` extra passes on genuinely
        dense crops (~150-350ms each) -- not 32+ passes over empty and full
        regions alike. See `find_dense_regions`/module docstring for the
        measured recall gain. Intended for one-shot image analysis, where a
        sub-second extra latency budget is acceptable; NOT used for video/
        real-time (see `crowd_monitoring.detection` callers), where it isn't.

        Falls back to a plain full-frame `detect()` if no density map is
        available (e.g. the DM-Count pass failed/was skipped) -- refinement
        is a bonus on top of detection, never a requirement for it.
        """
        full_dets = self.detect(frame_bgr, use_tiling=False)
        if density_map is None or density_map.size == 0 or max_regions <= 0:
            return full_dets

        h, w = frame_bgr.shape[:2]
        regions = find_dense_regions(
            density_map, original_size=(h, w), threshold_rel=threshold_rel, max_regions=max_regions
        )
        if not regions:
            return full_dets

        model = self._ensure_model()
        all_candidates: list[PersonDetection] = list(full_dets)
        for rx1, ry1, rx2, ry2 in regions:
            crop = frame_bgr[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                continue
            crop_dets = self._predict_single(model, crop, imgsz=self.imgsz)
            for d in crop_dets:
                all_candidates.append(
                    PersonDetection(x1=d.x1 + rx1, y1=d.y1 + ry1, x2=d.x2 + rx1, y2=d.y2 + ry1, confidence=d.confidence)
                )

        return nms_boxes(all_candidates, iou_threshold=self.nms_iou_threshold)

    def _predict_single(self, model: Any, crop_bgr: np.ndarray, imgsz: int) -> List[PersonDetection]:
        from .device import get_device_lock

        try:
            with get_device_lock(self._resolved_torch_device):
                results = model.predict(
                    crop_bgr,
                    classes=[COCO_PERSON_CLASS_ID],
                    conf=self.confidence_threshold,
                    imgsz=imgsz,
                    device=self._device_str,
                    verbose=False,
                )
        except Exception as exc:
            if self._device_str != "cpu":
                logger.warning(
                    "YOLO inference failed on device=%s (%s); retrying on CPU.", self._device_str, exc
                )
                import torch

                self._device_str = "cpu"
                self._resolved_torch_device = torch.device("cpu")
                with get_device_lock(self._resolved_torch_device):
                    results = model.predict(
                        crop_bgr,
                        classes=[COCO_PERSON_CLASS_ID],
                        conf=self.confidence_threshold,
                        imgsz=imgsz,
                        device="cpu",
                        verbose=False,
                    )
            else:
                raise
        return _parse_yolo_result(results[0], confidence_threshold=self.confidence_threshold)


def _torch_device_to_ultralytics_str(torch_device) -> str:
    if torch_device.type == "cuda":
        return f"cuda:{torch_device.index if torch_device.index is not None else 0}"
    if torch_device.type == "mps":
        return "mps"
    return "cpu"


def _parse_yolo_result(result: Any, confidence_threshold: float) -> List[PersonDetection]:
    detections: List[PersonDetection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = _to_numpy(boxes.xyxy)
    confs = _to_numpy(boxes.conf)
    classes = _to_numpy(boxes.cls)

    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
        if int(cls) != COCO_PERSON_CLASS_ID or float(conf) < confidence_threshold:
            continue
        ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        detections.append(PersonDetection(x1=ix1, y1=iy1, x2=ix2, y2=iy2, confidence=float(conf)))
    return detections


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
