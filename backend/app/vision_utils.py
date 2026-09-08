"""Shared helpers used by every entry point that renders or detects on raw
frames: image jobs, the video job runner, and real-time feeds.

Centralizing this avoids three slightly-different copies of "run the person
detector without fabricating a count" and "turn a BGR frame into a payload a
browser can render" -- both used identically whether the frame came from a
single uploaded image, a video job's frame loop, or a live feed's worker
thread.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Awaitable, Callable, List, Optional, Tuple

import cv2
import numpy as np

from crowd_monitoring.detection import PersonDetection, PersonDetector
from crowd_monitoring.overlay import draw_person_boxes

if TYPE_CHECKING:
    from crowd_monitoring.pipeline import FrameResult

    from .schemas.jobs import ImageVisualizations

logger = logging.getLogger(__name__)

# Localization points are rendered in this color everywhere (image
# visualizations and the "combined" view) so a point means the same thing
# across every panel.
_LOCALIZATION_COLOR_BGR = (0, 140, 255)


def resize_max_side(frame_bgr: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale (never upscale) so the longer side is at most `max_side`,
    preserving aspect ratio. No-op if the frame already fits."""
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame_bgr
    scale = max_side / float(longest)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_jpeg_data_url(frame_bgr: np.ndarray, quality: int = 85, max_side: Optional[int] = None) -> str:
    """Encode a BGR frame as a `data:image/jpeg;base64,...` URL a browser
    `<img>`/CSS background can use directly, with no server-side static file
    or extra request involved."""
    frame = resize_max_side(frame_bgr, max_side) if max_side else frame_bgr
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def detect_persons_sync(
    person_detector: PersonDetector,
    frame_bgr: np.ndarray,
    context: str,
) -> Tuple[List[PersonDetection], Optional[int]]:
    """Runs the (optional) person detector for one frame on the calling
    thread. Returns `(detections, visible_count)` -- `visible_count` is
    `None`, never a fake 0, whenever detection didn't actually run (model
    unavailable or it failed)."""
    if not person_detector.is_available:
        return [], None
    try:
        detections = person_detector.detect(frame_bgr)
        return detections, len(detections)
    except Exception:
        logger.exception("Person detector failed (%s)", context)
        return [], None


def _draw_localization_points(frame_bgr: np.ndarray, points_xy: np.ndarray, radius: int = 4) -> np.ndarray:
    out = frame_bgr.copy()
    for x, y in points_xy:
        cv2.circle(out, (int(round(float(x))), int(round(float(y)))), radius, _LOCALIZATION_COLOR_BGR, -1, lineType=cv2.LINE_AA)
    return out


def build_image_visualizations(
    frame_bgr: np.ndarray,
    result: "FrameResult",
    detections: List[PersonDetection],
    detector_available: bool,
    max_side: int = 1600,
    quality: int = 85,
) -> "ImageVisualizations":
    """Build the six separate image-page views the product requires:
    detection (real boxes, or omitted if the detector is unavailable --
    never a plain image mislabeled as a detection result), density (pure
    colorized density map, not a recolored original), heatmap (density
    blended onto the original), localization (density-map peaks marked on
    the original), and combined (heatmap + points + boxes together).

    All images are capped to `max_side` and JPEG-encoded as base64 data URLs
    so they can go straight into the JSON response with no extra request or
    server-side static file involved; aspect ratio is always preserved by
    `resize_max_side`.
    """
    from .schemas.jobs import ImageVisualizations

    points_xy = result.localization.points_xy
    density_img = result.heatmap_bgr
    heatmap_img = result.overlay_bgr
    localization_img = _draw_localization_points(frame_bgr, points_xy)

    combined_img = _draw_localization_points(heatmap_img, points_xy, radius=3)
    detection_img: Optional[np.ndarray] = None
    if detector_available:
        detection_img = draw_person_boxes(frame_bgr, detections)
        combined_img = draw_person_boxes(combined_img, detections)

    def enc(img: Optional[np.ndarray]) -> Optional[str]:
        return encode_jpeg_data_url(img, quality=quality, max_side=max_side) if img is not None else None

    return ImageVisualizations(
        detection=enc(detection_img),
        density=enc(density_img),
        heatmap=enc(heatmap_img),
        localization=enc(localization_img),
        combined=enc(combined_img),
    )


async def detect_persons_with_refinement_async(
    run_sync: Callable[..., Awaitable],
    person_detector: PersonDetector,
    frame_bgr: np.ndarray,
    density_map: Optional[np.ndarray],
    context: str,
) -> Tuple[List[PersonDetection], Optional[int]]:
    """Like `detect_persons_async`, but for one-shot image analysis: uses
    `PersonDetector.detect_with_density_refinement`, which adds selective,
    DM-Count-guided re-detection on genuinely dense regions on top of the
    full-frame pass (see `crowd_monitoring.detection` module docstring for the
    measured recall gain). Not used for video/real-time, where the extra
    latency isn't in budget -- those keep the plain full-frame `detect()`
    via `detect_persons_async` above."""
    if not person_detector.is_available:
        return [], None
    try:
        detections = await run_sync(person_detector.detect_with_density_refinement, frame_bgr, density_map)
        return detections, len(detections)
    except Exception:
        logger.exception("Person detector (density-refined) failed (%s)", context)
        return [], None


async def detect_persons_async(
    run_sync: Callable[..., Awaitable],
    person_detector: PersonDetector,
    frame_bgr: np.ndarray,
    context: str,
    detector_broken: bool = False,
) -> Tuple[List[PersonDetection], Optional[int], bool]:
    """Async wrapper around `detect_persons_sync` that runs the detector via
    `run_sync` (e.g. `PipelineRuntime.run_sync`, keeping it off the event
    loop) and additionally tracks a sticky `detector_broken` flag so a caller
    processing many frames (a whole video) doesn't re-attempt and re-log a
    detector that just failed, once it's already established the detector is
    broken for this run."""
    if detector_broken or not person_detector.is_available:
        return [], None, detector_broken
    try:
        detections = await run_sync(person_detector.detect, frame_bgr)
        return detections, len(detections), False
    except Exception:
        logger.exception(
            "Person detector failed (%s); visible_count will be unavailable for the rest of it", context
        )
        return [], None, True
