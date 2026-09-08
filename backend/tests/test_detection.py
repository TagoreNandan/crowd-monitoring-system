"""Unit tests for `crowd_monitoring.detection`: YOLO-output filtering (person-class,
confidence-thresholded, real bounding boxes) and `PersonDetector`'s lazy
load/availability/fallback plumbing. No real weights are loaded and no real
Ultralytics model runs -- `_parse_yolo_result` is tested directly against a
fake `Results`-shaped object, and `PersonDetector` is tested with its
`_model` swapped for a fake.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from crowd_monitoring.detection import (
    COCO_PERSON_CLASS_ID,
    PersonDetection,
    PersonDetector,
    PersonDetectorUnavailableError,
    _parse_yolo_result,
)

_DOG_CLASS_ID = 16  # COCO "dog" -- any non-person class works here.


class _FakeBoxesLen:
    """`SimpleNamespace` doesn't support `len()` via `__len__` as an
    attribute, so wrap in a tiny real class for the boxes container."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self):
        return len(self.cls)


def _fake_result(rows: list[tuple[float, float, float, float, float, int]]):
    if not rows:
        return SimpleNamespace(boxes=_FakeBoxesLen(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))))
    xyxy = np.array([[r[0], r[1], r[2], r[3]] for r in rows], dtype=np.float32)
    conf = np.array([r[4] for r in rows], dtype=np.float32)
    cls = np.array([r[5] for r in rows], dtype=np.float32)
    return SimpleNamespace(boxes=_FakeBoxesLen(xyxy, conf, cls))


def test_parse_yolo_result_keeps_only_person_above_threshold():
    result = _fake_result(
        [
            (10, 10, 40, 60, 0.95, COCO_PERSON_CLASS_ID),  # person, confident -> kept
            (20, 20, 30, 30, 0.10, COCO_PERSON_CLASS_ID),  # person, low confidence -> dropped
            (0, 0, 100, 100, 0.99, _DOG_CLASS_ID),  # confident but not a person -> dropped
        ]
    )

    detections = _parse_yolo_result(result, confidence_threshold=0.4)

    assert len(detections) == 1
    det = detections[0]
    assert isinstance(det, PersonDetection)
    assert det.confidence == pytest.approx(0.95, abs=1e-4)
    assert det.x1 == 10 and det.y1 == 10 and det.x2 == 40 and det.y2 == 60
    assert det.track_id is None


def test_parse_yolo_result_respects_configurable_confidence_threshold():
    result = _fake_result([(0, 0, 10, 10, 0.5, COCO_PERSON_CLASS_ID)])

    assert len(_parse_yolo_result(result, confidence_threshold=0.6)) == 0
    assert len(_parse_yolo_result(result, confidence_threshold=0.4)) == 1


def test_parse_yolo_result_never_returns_non_person_classes():
    non_person_class_ids = [i for i in range(0, 80) if i != COCO_PERSON_CLASS_ID]
    result = _fake_result([(0, 0, 10, 10, 0.99, cid) for cid in non_person_class_ids])

    assert _parse_yolo_result(result, confidence_threshold=0.1) == []


def test_parse_yolo_result_drops_degenerate_boxes():
    # A box with zero/negative area after rounding.
    result = _fake_result([(10, 10, 10, 10, 0.9, COCO_PERSON_CLASS_ID)])

    assert _parse_yolo_result(result, confidence_threshold=0.4) == []


def test_parse_yolo_result_with_no_boxes_returns_empty_list():
    result = _fake_result([])
    assert _parse_yolo_result(result, confidence_threshold=0.1) == []


class _FakeYoloModel:
    def __init__(self, result):
        self._result = result
        self.predict_calls: list[dict] = []

    def predict(self, frame, **kwargs):
        self.predict_calls.append(kwargs)
        return [self._result]


def test_person_detector_detect_uses_configured_threshold():
    detector = PersonDetector(confidence_threshold=0.5)
    detector._model = _FakeYoloModel(
        _fake_result(
            [
                (0, 0, 50, 50, 0.9, COCO_PERSON_CLASS_ID),
                (10, 10, 20, 20, 0.3, COCO_PERSON_CLASS_ID),
            ]
        )
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.9, abs=1e-4)


def test_person_detector_passes_person_only_class_filter_and_imgsz(monkeypatch):
    detector = PersonDetector(confidence_threshold=0.25, imgsz=960)
    fake_model = _FakeYoloModel(_fake_result([]))
    detector._model = fake_model
    detector._device_str = "cpu"

    detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert len(fake_model.predict_calls) == 1
    call = fake_model.predict_calls[0]
    assert call["classes"] == [COCO_PERSON_CLASS_ID]
    assert call["imgsz"] == 960
    assert call["device"] == "cpu"


def test_person_detector_is_available_false_when_ultralytics_missing(monkeypatch):
    import crowd_monitoring.detection as detection_module

    monkeypatch.setattr(detection_module, "_YOLO", None)
    detector = PersonDetector()

    assert detector.is_available is False
    with pytest.raises(PersonDetectorUnavailableError):
        detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))


def test_person_detector_marks_unavailable_after_load_failure(monkeypatch, tmp_path):
    import crowd_monitoring.detection as detection_module

    class _ExplodingYolo:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("simulated weight load failure")

    monkeypatch.setattr(detection_module, "_YOLO", _ExplodingYolo)
    detector = PersonDetector(weights_path=tmp_path / "does-not-exist.pt")

    assert detector.is_available is True  # not yet attempted
    with pytest.raises(PersonDetectorUnavailableError):
        detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert detector.is_available is False
    # A second call doesn't retry loading -- it fails fast.
    with pytest.raises(PersonDetectorUnavailableError):
        detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))


def test_person_detector_falls_back_to_cpu_on_device_inference_failure():
    detector = PersonDetector()
    detector._device_str = "mps"

    good_result = _fake_result([(0, 0, 10, 10, 0.9, COCO_PERSON_CLASS_ID)])

    class _FlakyModel:
        def __init__(self):
            self.calls = []

        def predict(self, frame, **kwargs):
            self.calls.append(kwargs["device"])
            if kwargs["device"] != "cpu":
                raise RuntimeError("simulated MPS op not supported")
            return [good_result]

    detector._model = _FlakyModel()
    detections = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detector._device_str == "cpu"
    assert detector._model.calls == ["mps", "cpu"]
