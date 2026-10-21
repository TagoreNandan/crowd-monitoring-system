"""Real-time feed orchestration: strict latest-frame-wins semantics with a
FAST path (person detection + tracking, runs against whatever frame is
current) fully decoupled from a SLOW path (DM-Count density/heatmap,
paced to ~1 Hz by default) -- neither ever blocks the other, and neither
ever builds a backlog of stale frames.

    SOURCE
      |
      v
    PRODUCER THREAD  --(overwrites)-->  LatestSlot[Frame]
                                          |            |
                              (reads latest)      (reads latest)
                                          v            v
                                   FAST THREAD    DENSITY THREAD
                                   (detector +     (DM-Count,
                                    tracker,        ~1/sec, paced)
                                    every frame)
                                          |            |
                                   on_fast_result  on_density_result

`LatestSlot` (see `latest_slot.py`) is what makes this work: `set()` never
blocks and never grows a queue, and each reader's `get()` only ever returns a
frame strictly newer than the last one it saw -- a slow reader (the density
thread) simply skips whatever frames arrived while it was busy, rather than
queueing them up or slowing the fast thread down.

This intentionally does NOT reuse `StreamSession`/`InferenceWorker`
(`session.py`/`worker.py`): those model a single producer -> single
consumer -> single `pipeline.analyze()` pipeline stage, which is exactly the
"DM-Count on the critical path" shape a live feed can't afford. Nothing here
duplicates their logic -- it solves a different problem (two independent
consumer cadences over one live source) that a single bounded queue can't
express.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import cv2

from ..detection import PersonDetection, PersonDetector
from ..pipeline import CrowdAnalysisPipeline, FrameResult
from ..tracking import EgoMotionEstimator, PersonTracker, TrackedPerson
from .frame_source import Frame, FrameSource, SourceMetadata
from .latest_slot import LatestSlot

logger = logging.getLogger(__name__)


@dataclass
class FastResult:
    """One person-detection/tracking pass over the current latest frame."""

    frame: Frame
    detections: List[TrackedPerson]
    visible_count: Optional[int]
    latency_ms: float
    frames_seen: int
    # False when the detector was skipped for this frame (detection_stride)
    # and `detections` are the last tracked boxes, held as-is.
    detected: bool = True


@dataclass
class DensityResult:
    """One DM-Count pass over whatever frame was current when it started."""

    frame: Frame
    result: FrameResult
    latency_ms: float
    updates: int


@dataclass
class RealtimeError:
    stage: str  # "producer", "fast", or "density"
    error: Exception


FastSink = Callable[[FastResult], None]
DensitySink = Callable[[DensityResult], None]
ErrorSink = Callable[[RealtimeError], None]


class RealtimeFeedSession:
    """Owns one live feed end to end: a producer thread reading `source`
    into a `LatestSlot`, an independent fast detector/tracker thread, and an
    independent, paced DM-Count thread. `on_fast_result`/`on_density_result`
    fire on their respective threads as soon as each result is ready --
    callers (e.g. the WebSocket broadcaster) must not do slow, blocking work
    in either callback, since that would extend that path's own cycle time
    (it will NOT block the other path, by construction).
    """

    def __init__(
        self,
        session_id: str,
        source: FrameSource,
        pipeline: CrowdAnalysisPipeline,
        person_detector: PersonDetector,
        tracker: Optional[PersonTracker] = None,
        on_fast_result: Optional[FastSink] = None,
        on_density_result: Optional[DensitySink] = None,
        on_error: Optional[ErrorSink] = None,
        density_hz: float = 1.0,
        loop_source: bool = True,
        ego_motion_compensation: bool = True,
        detection_stride: int = 1,
    ):
        self.session_id = session_id
        self._source = source
        self._pipeline = pipeline
        self._person_detector = person_detector
        self._tracker = tracker or PersonTracker()
        self._on_fast_result = on_fast_result
        self._on_density_result = on_density_result
        self._on_error = on_error
        self._density_interval_s = (1.0 / density_hz) if density_hz > 0 else 1.0
        self._loop_source = loop_source
        # Run the detector on every Nth source frame only; with many feeds
        # sharing one GPU this is what keeps the detector from saturating.
        self._detection_stride = max(1, int(detection_stride))
        # Background-only camera-motion estimate fed to the tracker so IDs
        # survive drone pans (see crowd_monitoring.tracking.EgoMotionEstimator).
        self._ego_motion: Optional[EgoMotionEstimator] = EgoMotionEstimator() if ego_motion_compensation else None

        self._latest_frame: LatestSlot[Frame] = LatestSlot()
        self._stop_event = threading.Event()
        self._producer_thread: Optional[threading.Thread] = None
        self._fast_thread: Optional[threading.Thread] = None
        self._density_thread: Optional[threading.Thread] = None

        self.status: str = "created"
        self.metadata: Optional[SourceMetadata] = None
        self.frames_produced = 0
        self.frames_detected = 0
        self.density_updates = 0
        self.error_count = 0
        self.started_at: Optional[float] = None
        self.stopped_at: Optional[float] = None
        self.latest_fast_result: Optional[FastResult] = None
        self.latest_density_result: Optional[DensityResult] = None

    @property
    def is_running(self) -> bool:
        return self.status == "running"

    @property
    def source(self) -> FrameSource:
        return self._source

    @property
    def source_status(self) -> str:
        """"LIVE", or a failover state ("BACKUP_ACTIVE"/"SIGNAL_LOST") when
        the source is a `FailoverFrameSource`; plain sources are always LIVE."""
        return getattr(self._source, "source_status", "LIVE")

    def start(self) -> SourceMetadata:
        if self.status == "running":
            assert self.metadata is not None
            return self.metadata

        self._stop_event.clear()
        self.metadata = self._source.open()
        self.started_at = time.time()
        self.status = "running"

        self._producer_thread = threading.Thread(
            target=self._run_producer, name=f"crowd-monitoring-rt-producer-{self.session_id}", daemon=True
        )
        self._fast_thread = threading.Thread(
            target=self._run_fast_loop, name=f"crowd-monitoring-rt-fast-{self.session_id}", daemon=True
        )
        self._density_thread = threading.Thread(
            target=self._run_density_loop, name=f"crowd-monitoring-rt-density-{self.session_id}", daemon=True
        )
        self._producer_thread.start()
        self._fast_thread.start()
        self._density_thread.start()
        logger.info(
            "RealtimeFeedSession %s started (source=%s, density_hz=%.2f)",
            self.session_id, self.metadata.description, 1.0 / self._density_interval_s,
        )
        return self.metadata

    def stop(self, timeout: float = 3.0) -> None:
        if self.status in ("stopped", "completed", "failed"):
            return
        logger.info("Stopping RealtimeFeedSession %s...", self.session_id)
        self._stop_event.set()
        self._latest_frame.close()
        for t in (self._producer_thread, self._fast_thread, self._density_thread):
            if t is not None and t.is_alive():
                t.join(timeout=timeout)
        try:
            self._source.close()
        except Exception:
            logger.exception("Error closing source for session %s", self.session_id)
        self.stopped_at = time.time()
        if self.status == "running":
            self.status = "stopped"
        logger.info("RealtimeFeedSession %s stopped.", self.session_id)

    # -- producer: continuously reads frames, the LatestSlot keeps only the newest --
    def _run_producer(self) -> None:
        try:
            start_wall = time.monotonic()
            start_ts: Optional[float] = None
            while not self._stop_event.is_set():
                for frame in self._source.frames():
                    if self._stop_event.is_set():
                        break

                    # Pace file-based sources to real-time playback speed if fps/timestamp is known
                    if frame.source_fps and frame.source_fps > 0:
                        if start_ts is None:
                            start_ts = frame.timestamp_s
                            start_wall = time.monotonic()
                        target_elapsed = frame.timestamp_s - start_ts
                        actual_elapsed = time.monotonic() - start_wall
                        sleep_needed = target_elapsed - actual_elapsed
                        if sleep_needed > 0.001:
                            time.sleep(min(sleep_needed, 0.5))

                    self._latest_frame.set(frame)
                    self.frames_produced += 1
                if not self._loop_source or self._stop_event.is_set():
                    break
                self._source.close()
                self._source.open()
                start_ts = None
            if not self._stop_event.is_set() and self.status == "running":
                self.status = "completed"
        except Exception as exc:
            logger.exception("Producer loop error in RealtimeFeedSession %s", self.session_id)
            self.error_count += 1
            self._emit_error("producer", exc)
        finally:
            self._latest_frame.close()

    # -- fast path: person detection + tracking against the current latest frame --
    def _should_detect(self, frame_number: int, last_detected: Optional[int]) -> bool:
        if last_detected is None or self._detection_stride <= 1:
            return True
        # A backwards jump means the source reopened/looped: detect right away.
        gap = frame_number - last_detected
        return gap < 0 or gap >= self._detection_stride

    def _run_fast_loop(self) -> None:
        last_version = 0
        max_fast_side = 1280
        last_detected_frame: Optional[int] = None
        while not self._stop_event.is_set():
            frame, last_version = self._latest_frame.get(last_version, timeout=0.5)
            if frame is None:
                if self._latest_frame.closed:
                    return
                continue
            previous = self.latest_fast_result
            if previous is not None and not self._should_detect(frame.frame_number, last_detected_frame):
                # Skipped frame: still deliver it (the UI stream stays smooth)
                # with the last tracked boxes and the last real latency.
                held = FastResult(
                    frame=frame,
                    detections=previous.detections,
                    visible_count=previous.visible_count,
                    latency_ms=previous.latency_ms,
                    frames_seen=self.frames_detected,
                    detected=False,
                )
                try:
                    if self._on_fast_result is not None:
                        self._on_fast_result(held)
                except Exception as exc:
                    logger.exception("Fast (held frame) callback error in RealtimeFeedSession %s", self.session_id)
                    self.error_count += 1
                    self._emit_error("fast", exc)
                continue
            last_detected_frame = frame.frame_number
            try:
                t0 = time.perf_counter()
                detections: List[PersonDetection]
                visible_count: Optional[int]
                if self._person_detector.is_available:
                    frame_bgr = frame.frame_bgr
                    fh, fw = frame_bgr.shape[:2]
                    if max(fh, fw) > max_fast_side:
                        scale = max_fast_side / float(max(fh, fw))
                        fast_w, fast_h = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
                        scaled_bgr = cv2.resize(frame_bgr, (fast_w, fast_h), interpolation=cv2.INTER_AREA)
                        raw_dets = self._person_detector.detect(scaled_bgr)
                        sx, sy = fw / float(fast_w), fh / float(fast_h)
                        detections = [
                            PersonDetection(
                                x1=int(round(d.x1 * sx)),
                                y1=int(round(d.y1 * sy)),
                                x2=int(round(d.x2 * sx)),
                                y2=int(round(d.y2 * sy)),
                                confidence=d.confidence,
                            )
                            for d in raw_dets
                        ]
                    else:
                        detections = self._person_detector.detect(frame_bgr)
                    visible_count = len(detections)
                else:
                    detections = []
                    visible_count = None
                motion = None
                if self._ego_motion is not None:
                    try:
                        motion = self._ego_motion.estimate(
                            frame.frame_bgr, [(d.x1, d.y1, d.x2, d.y2) for d in detections]
                        )
                    except Exception:
                        logger.exception("Ego-motion estimation failed in session %s", self.session_id)
                tracked = self._tracker.update(detections, motion=motion)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                self.frames_detected += 1
                result = FastResult(
                    frame=frame,
                    detections=tracked,
                    visible_count=visible_count,
                    latency_ms=latency_ms,
                    frames_seen=self.frames_detected,
                )
                self.latest_fast_result = result
                if self._on_fast_result is not None:
                    self._on_fast_result(result)
            except Exception as exc:
                logger.exception("Fast (detection) loop error in RealtimeFeedSession %s", self.session_id)
                self.error_count += 1
                self._emit_error("fast", exc)

    # -- slow path: DM-Count density/heatmap, paced to ~density_hz --
    def _run_density_loop(self) -> None:
        last_version = 0
        while not self._stop_event.is_set():
            frame, last_version = self._latest_frame.get(last_version, timeout=0.5)
            if frame is None:
                if self._latest_frame.closed:
                    return
                continue

            iter_start = time.perf_counter()
            try:
                result = self._pipeline.analyze(frame.frame_bgr)
                self.density_updates += 1
                density_result = DensityResult(
                    frame=frame,
                    result=result,
                    latency_ms=result.inference.latency_ms,
                    updates=self.density_updates,
                )
                self.latest_density_result = density_result
                if self._on_density_result is not None:
                    self._on_density_result(density_result)
            except Exception as exc:
                logger.exception("Density (DM-Count) loop error in RealtimeFeedSession %s", self.session_id)
                self.error_count += 1
                self._emit_error("density", exc)

            # Pace to ~density_hz regardless of how long analyze()+callback
            # actually took -- if that ever exceeds the interval, the loop
            # just runs flat-out (never blocks longer than necessary) rather
            # than drifting further behind.
            elapsed = time.perf_counter() - iter_start
            remaining = self._density_interval_s - elapsed
            if remaining > 0:
                self._stop_event.wait(timeout=remaining)

    def _emit_error(self, stage: str, exc: Exception) -> None:
        if self._on_error is not None:
            try:
                self._on_error(RealtimeError(stage=stage, error=exc))
            except Exception:
                logger.exception("Error in RealtimeFeedSession on_error callback for %s", self.session_id)
