"""MobileNet-SSD person detector.

This wraps OpenCV's `cv2.dnn` module around the classic MobileNet-SSD Caffe
model -- the same general-purpose object detector the original
Crowd-Cluster-Monitoring-Heatmap reference project uses (see the top-level
README's "reference-tier repos" section). It is a small, generic *detector*
used only to produce real, visible person bounding boxes (`visible_count`)
alongside DM-Count's separate dense `estimated_count`; it is never a
substitute for the density-regression pipeline and never fabricates a count
when it can't run.

Model files are intentionally NOT auto-downloaded here (unlike
`models/weights.py`'s DM-Count checkpoints, which come from one pinned,
already-vetted release asset): the classic MobileNet-SSD Caffe export is
mirrored by several different third-party repos with no single canonical,
verified URL, so guessing one to fetch a binary weight file from is not
something this module does. Instead it expects the two files to already be
present locally (see `weights/README.md` for where to get them) -- if
they're missing, `PersonDetector.is_available` is False and callers must
treat visible-person detection as unavailable for that run, not synthesize
a fake count.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = Path(os.environ.get("CROWDMON_WEIGHTS_DIR", REPO_ROOT / "weights"))

DEFAULT_PROTOTXT = Path(os.environ.get("CROWDMON_SSD_PROTOTXT", WEIGHTS_DIR / "MobileNetSSD_deploy.prototxt"))
DEFAULT_CAFFEMODEL = Path(os.environ.get("CROWDMON_SSD_CAFFEMODEL", WEIGHTS_DIR / "MobileNetSSD_deploy.caffemodel"))

DEFAULT_CONFIDENCE_THRESHOLD = 0.4

# The standard 21-class Pascal VOC label set the chuanqi305/MobileNet-SSD
# Caffe model was trained on. Detection is filtered on this exact index
# (not just "any confident detection"), so a confident car/dog/etc. never
# becomes a "person" box.
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
PERSON_CLASS_ID = VOC_CLASSES.index("person")

# MobileNet-SSD's native input resolution.
_INPUT_SIZE = (300, 300)


@dataclass
class PersonDetection:
    """One real, above-threshold, person-only bounding box. Coordinates are
    pixel coordinates in the original (un-resized) frame."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class PersonDetectorUnavailableError(RuntimeError):
    """Raised when the MobileNet-SSD model files can't be found. Callers
    must not catch this to invent a detection result -- treat visible-person
    detection as unavailable instead."""


class PersonDetector:
    """Loads the MobileNet-SSD Caffe model once (lazily, on first `detect()`
    call) and runs person-only detection on BGR frames. Mirrors
    `CrowdCounter`'s lazy-load-once shape, but is a wholly separate model:
    nothing here feeds into or reads from `CrowdAnalysisPipeline`."""

    def __init__(
        self,
        prototxt_path: Path = DEFAULT_PROTOTXT,
        caffemodel_path: Path = DEFAULT_CAFFEMODEL,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self.prototxt_path = Path(prototxt_path)
        self.caffemodel_path = Path(caffemodel_path)
        self.confidence_threshold = confidence_threshold
        self._net = None

    @property
    def is_available(self) -> bool:
        """Whether the model files exist locally. Does not attempt to load
        them -- a cheap, side-effect-free check callers can use to decide
        whether to skip detection entirely for a run."""
        return self.prototxt_path.exists() and self.caffemodel_path.exists()

    def _ensure_net(self):
        if self._net is None:
            if not self.is_available:
                raise PersonDetectorUnavailableError(
                    f"MobileNet-SSD model files not found (looked for "
                    f"'{self.prototxt_path}' and '{self.caffemodel_path}'). "
                    "See weights/README.md for where to get the prototxt + "
                    "caffemodel, or point CROWDMON_SSD_PROTOTXT / "
                    "CROWDMON_SSD_CAFFEMODEL at your own copies."
                )
            logger.info("Loading MobileNet-SSD person detector from %s", self.caffemodel_path)
            self._net = cv2.dnn.readNetFromCaffe(str(self.prototxt_path), str(self.caffemodel_path))
        return self._net

    def detect(self, frame_bgr: np.ndarray) -> List[PersonDetection]:
        """Run one forward pass and return real, person-only bounding boxes
        with confidence >= `confidence_threshold`. Every detection is read
        straight from the model's own output -- nothing here is synthesized
        or estimated."""
        net = self._ensure_net()
        return _filter_person_detections(
            net=net,
            frame_bgr=frame_bgr,
            confidence_threshold=self.confidence_threshold,
        )


def _filter_person_detections(net, frame_bgr: np.ndarray, confidence_threshold: float) -> List[PersonDetection]:
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, _INPUT_SIZE), 0.007843, _INPUT_SIZE, 127.5)
    net.setInput(blob)
    raw = net.forward()
    return parse_ssd_output(raw, frame_width=w, frame_height=h, confidence_threshold=confidence_threshold)


def parse_ssd_output(
    raw: np.ndarray,
    frame_width: int,
    frame_height: int,
    confidence_threshold: float,
) -> List[PersonDetection]:
    """Turn one MobileNet-SSD `net.forward()` output blob (shape
    `(1, 1, N, 7)`, rows `[image_id, class_id, confidence, x1, y1, x2, y2]`
    with box coordinates normalized to [0, 1]) into person-only, above-
    threshold, pixel-space boxes.

    Pulled out of `PersonDetector.detect()` so the filtering logic itself is
    directly unit-testable against a synthetic blob, without a real model or
    weight files.
    """
    detections: List[PersonDetection] = []
    for i in range(raw.shape[2]):
        confidence = float(raw[0, 0, i, 2])
        if confidence < confidence_threshold:
            continue
        class_id = int(raw[0, 0, i, 1])
        if class_id != PERSON_CLASS_ID:
            continue

        box = raw[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
        x1, y1, x2, y2 = box.astype(int).tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width - 1, x2), min(frame_height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(PersonDetection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence))
    return detections
