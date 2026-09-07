"""Unit tests for `crowdmon.detection`: SSD-output filtering (person-only,
confidence-thresholded, real bounding boxes) and `PersonDetector.detect()`'s
plumbing to it. No real Caffe model/weights are loaded -- the filtering
logic is tested directly against a synthetic `net.forward()`-shaped blob,
and `PersonDetector` is tested with its `_net` swapped for a fake.
"""

from __future__ import annotations

import numpy as np
import pytest

from crowdmon.detection import (
    PERSON_CLASS_ID,
    VOC_CLASSES,
    PersonDetection,
    PersonDetector,
    PersonDetectorUnavailableError,
    parse_ssd_output,
)

_CAR_CLASS_ID = 7  # "car" in the VOC label set -- any non-person class works here.


def _ssd_row(class_id: float, confidence: float, box=(0.1, 0.1, 0.5, 0.5)) -> list:
    x1, y1, x2, y2 = box
    return [0.0, class_id, confidence, x1, y1, x2, y2]


def _ssd_blob(rows: list[list[float]]) -> np.ndarray:
    arr = np.zeros((1, 1, len(rows), 7), dtype=np.float32)
    for i, row in enumerate(rows):
        arr[0, 0, i, :] = row
    return arr


def test_parse_ssd_output_keeps_only_person_above_threshold():
    raw = _ssd_blob(
        [
            _ssd_row(PERSON_CLASS_ID, 0.95, box=(0.1, 0.1, 0.4, 0.6)),  # person, confident -> kept
            _ssd_row(PERSON_CLASS_ID, 0.10, box=(0.2, 0.2, 0.3, 0.3)),  # person, low confidence -> dropped
            _ssd_row(_CAR_CLASS_ID, 0.99, box=(0.0, 0.0, 1.0, 1.0)),  # confident but not a person -> dropped
        ]
    )

    detections = parse_ssd_output(raw, frame_width=200, frame_height=100, confidence_threshold=0.4)

    assert len(detections) == 1
    det = detections[0]
    assert isinstance(det, PersonDetection)
    assert det.confidence == pytest.approx(0.95)
    # box normalized [0.1, 0.1, 0.4, 0.6] * (200, 100, 200, 100)
    assert det.x1 == 20 and det.y1 == 10
    assert det.x2 == 80 and det.y2 == 60


def test_parse_ssd_output_respects_configurable_confidence_threshold():
    raw = _ssd_blob([_ssd_row(PERSON_CLASS_ID, 0.5)])

    assert len(parse_ssd_output(raw, 100, 100, confidence_threshold=0.6)) == 0
    assert len(parse_ssd_output(raw, 100, 100, confidence_threshold=0.4)) == 1


def test_parse_ssd_output_never_returns_non_person_classes():
    non_person_class_ids = [i for i in range(len(VOC_CLASSES)) if i != PERSON_CLASS_ID]
    raw = _ssd_blob([_ssd_row(class_id, 0.99) for class_id in non_person_class_ids])

    detections = parse_ssd_output(raw, 100, 100, confidence_threshold=0.1)

    assert detections == []


def test_parse_ssd_output_clips_boxes_to_frame_bounds():
    # A box that would extend past the frame edge after denormalizing.
    raw = _ssd_blob([_ssd_row(PERSON_CLASS_ID, 0.9, box=(-0.1, -0.1, 1.2, 1.2))])

    detections = parse_ssd_output(raw, frame_width=100, frame_height=50, confidence_threshold=0.4)

    assert len(detections) == 1
    det = detections[0]
    assert 0 <= det.x1 < det.x2 <= 99
    assert 0 <= det.y1 < det.y2 <= 49


class _FakeNet:
    def __init__(self, raw: np.ndarray):
        self._raw = raw
        self.input_set = False

    def setInput(self, blob):
        self.input_set = True

    def forward(self):
        return self._raw


def test_person_detector_detect_uses_configured_threshold():
    detector = PersonDetector(confidence_threshold=0.5)
    detector._net = _FakeNet(
        _ssd_blob(
            [
                _ssd_row(PERSON_CLASS_ID, 0.9, box=(0.0, 0.0, 0.5, 0.5)),
                _ssd_row(PERSON_CLASS_ID, 0.3, box=(0.1, 0.1, 0.2, 0.2)),
            ]
        )
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.9)


def test_person_detector_raises_when_model_files_missing(tmp_path):
    detector = PersonDetector(
        prototxt_path=tmp_path / "does-not-exist.prototxt",
        caffemodel_path=tmp_path / "does-not-exist.caffemodel",
    )

    assert detector.is_available is False
    with pytest.raises(PersonDetectorUnavailableError):
        detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))
