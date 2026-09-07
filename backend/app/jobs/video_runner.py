"""Background runner for video jobs: owns the frame-by-frame loop that
Phase 1's `POST /api/jobs/video` deliberately left unimplemented.

Runs entirely off the asyncio event loop -- every OpenCV call, every
`PipelineRuntime.analyze()` call, and every `PersonDetector.detect()` call
goes through `PipelineRuntime`'s existing single-worker executor (see
`pipeline_runtime.py`), so a multi-minute video never blocks WebSocket
heartbeats or other requests. No new model or inference logic is
constructed here beyond the separate, optional `PersonDetector` (see
`crowdmon.detection`); frame-level bookkeeping (peak count, growth-ratio
trend) reuses `crowdmon.video.StreamState`, the same helper the CLI's video
loop already uses.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, List, Optional

import cv2

from crowdmon.detection import PersonDetection, PersonDetector
from crowdmon.overlay import render_video_overlay
from crowdmon.streaming import FileVideoSource
from crowdmon.tracking import PersonTracker
from crowdmon.video import StreamState
from crowdmon.video_encoding import ensure_browser_compatible_video

from ..pipeline_runtime import PipelineRuntime
from ..schemas.jobs import VideoFrameMetrics, VideoJobSummary
from ..schemas.ws import ErrorData, JobStartedData, WSEnvelope, WSEventType
from ..ws.manager import ConnectionManager
from .store import JobStore

logger = logging.getLogger(__name__)

# ~6-7 messages/sec -- inside the 5-10/s target, so a long video's progress
# stream never floods the socket.
PROGRESS_EMIT_INTERVAL_S = 0.15

# Fallback used only if the source video reports no/invalid FPS -- same
# fallback value scripts/run_video.py already uses for its own writer.
_DEFAULT_OUTPUT_FPS = 20.0

# Holds in-flight job tasks purely so they can't be garbage-collected
# mid-run (the standard asyncio idiom for a `create_task()` result nothing
# awaits) -- entries are removed by `_on_task_done` as each task finishes.
_background_tasks: set[asyncio.Task] = set()


def schedule_video_job(
    job_id: str,
    video_path: Path,
    runtime: PipelineRuntime,
    job_store: JobStore,
    connection_manager: ConnectionManager,
    person_detector: PersonDetector,
) -> asyncio.Task:
    """Fire-and-forget `run_video_job` onto the running event loop and
    return immediately -- called from the request handler so `POST
    /api/jobs/video` never waits on video processing."""
    task = asyncio.create_task(
        run_video_job(job_id, video_path, runtime, job_store, connection_manager, person_detector)
    )
    _background_tasks.add(task)
    task.add_done_callback(_on_task_done)
    return task


def _on_task_done(task: asyncio.Task) -> None:
    _background_tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        # `run_video_job` catches everything it can and marks the job
        # failed itself; this is a last-resort net so a bug in that handling
        # still surfaces as a logged error instead of an unhandled task
        # exception.
        logger.error("Unhandled exception in video job task", exc_info=exc)


def _build_metrics(
    frame_number: int,
    total_frames: Optional[int],
    elapsed_s: float,
    result: Any,
    peak_count: float,
    visible_count: Optional[int],
) -> VideoFrameMetrics:
    density = float(result.risk.components.get("relative_density_per_100px2", 0.0) or 0.0)
    latency_ms = float(result.inference.latency_ms)
    return VideoFrameMetrics(
        frame_number=frame_number,
        total_frames=total_frames,
        elapsed_s=round(elapsed_s, 3),
        visible_count=visible_count,
        estimated_count=round(float(result.inference.count), 2),
        peak_count=round(float(peak_count), 2),
        density=round(density, 3),
        cluster_count=int(result.clusters.n_clusters),
        largest_cluster=int(result.clusters.largest_cluster_size),
        risk_status=str(result.risk.status),
        risk_score=round(float(result.risk.score), 2),
        latency_ms=round(latency_ms, 2),
        fps=round(1000.0 / latency_ms, 2) if latency_ms > 0 else None,
    )


async def _detect_persons(
    runtime: PipelineRuntime,
    person_detector: PersonDetector,
    frame,
    job_id: str,
    detector_broken: bool,
) -> tuple[List[PersonDetection], Optional[int], bool]:
    """Runs the (optional) person detector for one frame. Returns
    `(detections, visible_count, detector_broken)` -- `visible_count` is
    `None`, never a fake 0, whenever detection didn't actually run (model
    files missing, or it failed once already this job)."""
    if detector_broken or not person_detector.is_available:
        return [], None, detector_broken
    try:
        detections = await runtime.run_sync(person_detector.detect, frame)
        return detections, len(detections), detector_broken
    except Exception:
        logger.exception(
            "Person detector failed for job %s; visible_count will be unavailable for the rest of this job", job_id
        )
        return [], None, True


async def run_video_job(
    job_id: str,
    video_path: Path,
    runtime: PipelineRuntime,
    job_store: JobStore,
    connection_manager: ConnectionManager,
    person_detector: PersonDetector,
) -> None:
    """Process one video job start to finish: open it, analyze every frame
    through the existing pipeline (plus the separate, optional person
    detector for `visible_count`), publish typed progress over the job's
    WebSocket channel, and always end in `completed` or `failed` -- never
    left `running` -- with all temp files removed either way.

    The annotated (boxes + stats panel) frame is written straight to a
    scratch output video one frame at a time via `cv2.VideoWriter` -- it is
    never sent over the WebSocket and never collected into a list, so a
    long video's memory footprint stays flat. Nothing in this phase serves
    or retains that annotated video afterwards (no download endpoint was
    requested), so it's deleted alongside the input upload once the job
    ends; the point of writing it here is to exercise the real overlay
    end-to-end, not to leave a dangling artifact.
    """
    channel = connection_manager.job_channel(job_id)
    record = job_store.get(job_id)
    source_description = record.source_description if record is not None else ""

    source: Optional[FileVideoSource] = None
    annotated_writer = None
    annotated_path = video_path.with_name(f"{video_path.stem}_annotated.mp4")
    started = time.time()
    state = StreamState()
    tracker = PersonTracker()
    frames_processed = 0
    total_frames: Optional[int] = None
    latest_metrics: Optional[VideoFrameMetrics] = None
    last_emit = 0.0
    detector_broken = False

    if not person_detector.is_available:
        logger.info("Person detector model files not found; job %s will report visible_count=null", job_id)

    try:
        job_store.mark_running(job_id)
        await connection_manager.broadcast(
            channel,
            WSEnvelope(
                type=WSEventType.JOB_STARTED,
                job_id=job_id,
                data=JobStartedData(job_type="video", source_description=source_description).model_dump(),
            ),
        )

        source = FileVideoSource(video_path)
        try:
            metadata = await runtime.run_sync(source.open)
        except RuntimeError as exc:
            raise RuntimeError(f"Could not open uploaded video '{video_path.name}': {exc}") from exc

        total_frames = metadata.total_frames
        output_fps = metadata.fps if metadata.fps else _DEFAULT_OUTPUT_FPS

        frame_iter = await runtime.run_sync(source.frames)
        while True:
            frame = await runtime.run_sync(next, frame_iter, None)
            if frame is None:
                break

            result = await runtime.analyze(frame.frame_bgr, growth_ratio=state.growth_ratio())
            state.update(result.inference.count)
            frames_processed += 1

            detections, visible_count, detector_broken = await _detect_persons(
                runtime, person_detector, frame.frame_bgr, job_id, detector_broken
            )
            tracked_boxes = tracker.update(detections) if detections else []

            latest_metrics = _build_metrics(
                frame_number=frames_processed,
                total_frames=total_frames,
                elapsed_s=time.time() - started,
                result=result,
                peak_count=state.peak_count,
                visible_count=visible_count,
            )

            annotated = render_video_overlay(
                frame.frame_bgr,
                tracked_boxes if detections else detections,
                **latest_metrics.model_dump(),
            )
            if annotated_writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                annotated_writer = await runtime.run_sync(
                    cv2.VideoWriter, str(annotated_path), fourcc, output_fps, (w, h)
                )
            await runtime.run_sync(annotated_writer.write, annotated)

            progress = (frames_processed / total_frames) if total_frames else 0.0
            job_store.update_video_progress(
                job_id,
                progress=progress,
                current_frame=frames_processed,
                total_frames=total_frames,
                peak_count=state.peak_count,
                metrics=latest_metrics.model_dump(),
            )

            now = time.monotonic()
            if now - last_emit >= PROGRESS_EMIT_INTERVAL_S:
                await connection_manager.broadcast(
                    channel,
                    WSEnvelope(type=WSEventType.JOB_PROGRESS, job_id=job_id, data=latest_metrics.model_dump()),
                )
                last_emit = now

        if frames_processed == 0 or latest_metrics is None:
            raise RuntimeError("No frames could be decoded from the uploaded video.")

        if annotated_writer is not None:
            await runtime.run_sync(annotated_writer.release)
            annotated_writer = None
            if annotated_path.exists():
                try:
                    await runtime.run_sync(ensure_browser_compatible_video, annotated_path)
                except Exception:
                    logger.exception("Failed to transcode annotated video for job %s", job_id)

        summary = VideoJobSummary(
            frames_processed=frames_processed,
            total_frames=total_frames,
            elapsed_s=round(time.time() - started, 3),
            peak_count=round(state.peak_count, 2),
            final_metrics=latest_metrics,
        )
        job_store.mark_completed(job_id, summary.model_dump())
        await connection_manager.broadcast(
            channel,
            WSEnvelope(type=WSEventType.JOB_COMPLETED, job_id=job_id, data=summary.model_dump()),
        )

    except Exception as exc:
        logger.exception("Video job %s failed", job_id)
        job_store.mark_failed(job_id, str(exc))
        try:
            await connection_manager.broadcast(
                channel,
                WSEnvelope(
                    type=WSEventType.JOB_ERROR,
                    job_id=job_id,
                    data=ErrorData(code="video_processing_failed", message=str(exc)).model_dump(),
                ),
            )
        except Exception:
            logger.exception("Failed to publish job.error for job %s", job_id)
    finally:
        if source is not None:
            try:
                await runtime.run_sync(source.close)
            except Exception:
                logger.exception("Failed to release VideoCapture for job %s", job_id)
        if annotated_writer is not None:
            try:
                await runtime.run_sync(annotated_writer.release)
            except Exception:
                logger.exception("Failed to release annotated VideoWriter for job %s", job_id)
        try:
            video_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove temp video for job %s", job_id, exc_info=True)
        try:
            annotated_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove annotated temp video for job %s", job_id, exc_info=True)
