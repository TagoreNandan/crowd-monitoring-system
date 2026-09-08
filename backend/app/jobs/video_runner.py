"""Background runner for video jobs: owns the frame-by-frame loop that
Phase 1's `POST /api/jobs/video` deliberately left unimplemented.

Runs entirely off the asyncio event loop -- every OpenCV call, every
`PipelineRuntime.analyze()` call, and every `PersonDetector.detect()` call
goes through `PipelineRuntime`'s existing single-worker executor (see
`pipeline_runtime.py`), so a multi-minute video never blocks WebSocket
heartbeats or other requests.

Performance architecture (the fix for videos that used to take minutes):
DM-Count (`runtime.analyze`, the dense density/heatmap/localization/risk
pass) is expensive -- roughly half a second per frame on this project's own
Apple MPS hardware -- and the CrowdHuman-trained YOLOv8n person detector,
while much cheaper, is still ~85-180ms/frame at the resolution needed for
real recall on a dense/4K frame (see `crowd_monitoring.detection`). Running either
of these on every one of a 60fps source's frames is where the old
multi-minute runtime went. So both are decoupled from the decode/write loop,
each at its own honest, time-based cadence:

  - the person detector + tracker run at `settings.video_detection_hz`
    times/sec of *source* video (default 20/sec -- far smoother than the eye
    needs, well below 60fps). Frames between detector passes reuse the
    tracker's last known boxes AS-IS (held in place, not re-detected) rather
    than pretending a fresh detection happened on every frame;
  - the DM-Count pass runs at `settings.density_analysis_hz` times/sec
    (default 1/sec), and its result (count, density map, heatmap,
    localization, clusters, risk) is cached and reused for the frames in
    between.

Every decoded frame is still written to the annotated output video at the
source frame rate, so playback stays smooth -- only the detector and
density numbers refresh periodically, at rates chosen because a person's
visible position and the crowd's density don't meaningfully change
frame-to-frame at 60fps anyway. `VideoJobSummary.frames_detected` and the
cadence values logged at job start report what actually happened -- this
module never claims a skipped frame was freshly analyzed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

import cv2

from crowd_monitoring.detection import PersonDetector
from crowd_monitoring.overlay import draw_density_heatmap_overlay, draw_stats_panel, render_video_overlay
from crowd_monitoring.streaming import FileVideoSource
from crowd_monitoring.tracking import PersonTracker
from crowd_monitoring.video import StreamState
from crowd_monitoring.video_encoding import ensure_browser_compatible_video

from ..pipeline_runtime import PipelineRuntime
from ..schemas.jobs import VideoFrameMetrics, VideoJobSummary
from ..schemas.ws import ErrorData, JobStartedData, WSEnvelope, WSEventType
from ..settings import get_settings
from ..vision_utils import detect_persons_async, encode_jpeg_data_url
from ..ws.manager import ConnectionManager
from .store import JobStore

logger = logging.getLogger(__name__)

# ~6-7 messages/sec -- inside the 5-10/s target, so a long video's progress
# stream never floods the socket.
PROGRESS_EMIT_INTERVAL_S = 0.15

# Fallback used only if the source video reports no/invalid FPS -- same
# fallback value scripts/run_video.py already uses for its own writer.
_DEFAULT_OUTPUT_FPS = 20.0

# Fallback cadence (in frames) for the DM-Count density pass, used only if
# the source doesn't report an FPS to compute a time-based cadence from.
_DEFAULT_DENSITY_CADENCE_FRAMES = 15

# Same fallback idea for the person detector's cadence.
_DEFAULT_DETECTOR_CADENCE_FRAMES = 2

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
    density_analysis_hz: float = 1.0,
    video_detection_hz: float = 20.0,
) -> asyncio.Task:
    """Fire-and-forget `run_video_job` onto the running event loop and
    return immediately -- called from the request handler so `POST
    /api/jobs/video` never waits on video processing."""
    task = asyncio.create_task(
        run_video_job(
            job_id, video_path, runtime, job_store, connection_manager, person_detector,
            density_analysis_hz, video_detection_hz,
        )
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


def _density_cadence_frames(source_fps: Optional[float], density_analysis_hz: float) -> int:
    """How many frames to advance between DM-Count passes. Time-based (a
    fraction of a second of *source* video), not a fixed frame count, so a
    30fps and a 120fps source both get DM-Count refreshed at roughly the same
    real-world rate."""
    if not source_fps or source_fps <= 0 or density_analysis_hz <= 0:
        return _DEFAULT_DENSITY_CADENCE_FRAMES
    return max(1, round(source_fps / density_analysis_hz))


def _detector_cadence_frames(source_fps: Optional[float], video_detection_hz: float) -> int:
    """How many frames to advance between person-detector passes. Same
    time-based idea as `_density_cadence_frames`, at a much higher target
    rate since the detector is far cheaper than DM-Count."""
    if not source_fps or source_fps <= 0 or video_detection_hz <= 0:
        return _DEFAULT_DETECTOR_CADENCE_FRAMES
    return max(1, round(source_fps / video_detection_hz))


def _build_metrics(
    frame_number: int,
    total_frames: Optional[int],
    elapsed_s: float,
    video_time_s: float,
    total_duration_s: Optional[float],
    result: Any,
    peak_count: float,
    visible_count: Optional[int],
) -> VideoFrameMetrics:
    """The ONE authoritative per-frame metrics object for a video job.
    Every consumer -- the job store (`latest_metrics`), the job.progress
    WebSocket broadcast, and the burned-in video overlay -- reads its values
    from this same object, never recomputing or re-deriving anything
    independently. See `VideoFrameMetrics` for why `elapsed_s` (wall-clock)
    and `video_time_s` (the video's own timestamp) are two distinct fields."""
    density = float(result.risk.components.get("relative_density_per_100px2", 0.0) or 0.0)
    latency_ms = float(result.inference.latency_ms)
    return VideoFrameMetrics(
        frame_number=frame_number,
        total_frames=total_frames,
        elapsed_s=round(elapsed_s, 3),
        video_time_s=round(video_time_s, 3),
        total_duration_s=round(total_duration_s, 3) if total_duration_s is not None else None,
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


async def run_video_job(
    job_id: str,
    video_path: Path,
    runtime: PipelineRuntime,
    job_store: JobStore,
    connection_manager: ConnectionManager,
    person_detector: PersonDetector,
    density_analysis_hz: float = 1.0,
    video_detection_hz: float = 20.0,
) -> None:
    """Process one video job start to finish: open it, run the person
    detector + tracker on every frame, run DM-Count periodically (see module
    docstring), publish typed progress over the job's WebSocket channel, and
    always end in `completed` or `failed` -- never left `running` -- with the
    temp *input* file always removed.

    The annotated (boxes + stats panel) output video IS retained on success
    (served by `GET /api/jobs/{job_id}/video`) -- unlike Phase 2A, this phase
    needs the browser to actually be able to play back the real per-frame
    result, not just read numbers from a WebSocket. It's deleted on failure
    (nothing valid to serve) and by `JobStore`'s normal TTL eviction
    otherwise.
    """
    settings = get_settings()
    channel = connection_manager.job_channel(job_id)
    record = job_store.get(job_id)
    source_description = record.source_description if record is not None else ""

    source: Optional[FileVideoSource] = None
    annotated_writer = None
    heatmap_writer = None
    combined_writer = None
    annotated_path = video_path.with_name(f"{video_path.stem}_annotated.mp4")
    heatmap_path = video_path.with_name(f"{video_path.stem}_heatmap.mp4")
    combined_path = video_path.with_name(f"{video_path.stem}_combined.mp4")
    started = time.time()
    state = StreamState()
    tracker = PersonTracker()
    frames_processed = 0
    frames_detected = 0
    density_updates = 0
    total_frames: Optional[int] = None
    latest_metrics: Optional[VideoFrameMetrics] = None
    latest_heavy_result: Any = None
    tracked_boxes: list = []
    visible_count: Optional[int] = None
    last_emit = 0.0
    detector_broken = False
    succeeded = False

    if not person_detector.is_available:
        logger.info("Person detector model unavailable; job %s will report visible_count=null", job_id)

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
        total_duration_s = (total_frames / metadata.fps) if (total_frames and metadata.fps) else None
        density_cadence = _density_cadence_frames(metadata.fps, density_analysis_hz)
        detector_cadence = _detector_cadence_frames(metadata.fps, video_detection_hz)
        logger.info(
            "Video job %s: source_fps=%s total_frames=%s density_cadence=%d frames (~%.1f Hz) "
            "detector_cadence=%d frames (~%.1f Hz)",
            job_id, metadata.fps, total_frames, density_cadence, density_analysis_hz,
            detector_cadence, video_detection_hz,
        )

        frame_iter = await runtime.run_sync(source.frames)
        while True:
            frame = await runtime.run_sync(next, frame_iter, None)
            if frame is None:
                break
            frames_processed += 1

            # Periodic: person detection + tracking. Runs at
            # `video_detection_hz` (default 20/sec), not every frame -- see
            # module docstring. Frames in between simply keep the tracker's
            # last known boxes (held in place -- the tracker itself isn't
            # touched, so no track ages out just because this frame skipped
            # detection); `frames_detected` counts only real detector calls.
            is_detector_frame = (frames_processed - 1) % detector_cadence == 0
            if is_detector_frame or frames_detected == 0:
                detections, visible_count, detector_broken = await detect_persons_async(
                    runtime.run_sync, person_detector, frame.frame_bgr, f"video job {job_id}", detector_broken
                )
                tracked_boxes = tracker.update(detections)
                frames_detected += 1

            # Expensive, periodic: DM-Count density/heatmap/localization/risk.
            # Always runs on the first frame so `latest_heavy_result` is never
            # None once the loop below needs it.
            is_density_frame = (frames_processed - 1) % density_cadence == 0
            if is_density_frame or latest_heavy_result is None:
                latest_heavy_result = await runtime.analyze(frame.frame_bgr, growth_ratio=state.growth_ratio())
                state.update(latest_heavy_result.inference.count)
                density_updates += 1

            latest_metrics = _build_metrics(
                frame_number=frames_processed,
                total_frames=total_frames,
                elapsed_s=time.time() - started,
                video_time_s=frame.timestamp_s,
                total_duration_s=total_duration_s,
                result=latest_heavy_result,
                peak_count=state.peak_count,
                visible_count=visible_count,
            )

            # The overlay is rendered directly from `latest_metrics` --
            # every value below is `latest_metrics.<field>`, nothing
            # recomputed or independently rounded -- so the burned-in video
            # and the UI/API can never show a different number for the same
            # frame (see `VideoFrameMetrics`/`draw_stats_panel` docstrings).
            annotated = render_video_overlay(
                frame.frame_bgr,
                tracked_boxes,
                frame_number=latest_metrics.frame_number,
                total_frames=latest_metrics.total_frames,
                video_time_s=latest_metrics.video_time_s,
                total_duration_s=latest_metrics.total_duration_s,
                visible_count=latest_metrics.visible_count,
                estimated_count=latest_metrics.estimated_count,
                peak_count=latest_metrics.peak_count,
                fps=latest_metrics.fps,
                latency_ms=latest_metrics.latency_ms,
                density=latest_metrics.density,
                cluster_count=latest_metrics.cluster_count,
                largest_cluster=latest_metrics.largest_cluster,
                risk_status=latest_metrics.risk_status,
                risk_score=latest_metrics.risk_score,
            )
            if annotated_writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                annotated_writer = await runtime.run_sync(
                    cv2.VideoWriter, str(annotated_path), fourcc, output_fps, (w, h)
                )
                heatmap_writer = await runtime.run_sync(
                    cv2.VideoWriter, str(heatmap_path), fourcc, output_fps, (w, h)
                )
                combined_writer = await runtime.run_sync(
                    cv2.VideoWriter, str(combined_path), fourcc, output_fps, (w, h)
                )
            await runtime.run_sync(annotated_writer.write, annotated)

            # Heatmap/Combined: reuse the ALREADY-COMPUTED density map for
            # this frame (`latest_heavy_result`, refreshed at the density
            # cadence above, cached in between) -- this never triggers a new
            # DM-Count call. The density-blend itself (`draw_density_heatmap_overlay`)
            # is the one non-trivial cost here (~20ms at full 4K); drawing
            # the stats panel/boxes on top of it is cheap, so it's computed
            # once and reused for both the pure-heatmap and combined frames.
            heatmap_base = await runtime.run_sync(
                draw_density_heatmap_overlay, frame.frame_bgr, latest_heavy_result.inference.density_map, 0.55
            )
            heatmap_frame = await runtime.run_sync(
                draw_stats_panel,
                heatmap_base,
                frame_number=latest_metrics.frame_number,
                total_frames=latest_metrics.total_frames,
                video_time_s=latest_metrics.video_time_s,
                total_duration_s=latest_metrics.total_duration_s,
                visible_count=latest_metrics.visible_count,
                estimated_count=latest_metrics.estimated_count,
                peak_count=latest_metrics.peak_count,
                fps=latest_metrics.fps,
                latency_ms=latest_metrics.latency_ms,
                density=latest_metrics.density,
                cluster_count=latest_metrics.cluster_count,
                largest_cluster=latest_metrics.largest_cluster,
                risk_status=latest_metrics.risk_status,
                risk_score=latest_metrics.risk_score,
            )
            await runtime.run_sync(heatmap_writer.write, heatmap_frame)

            combined_frame = await runtime.run_sync(render_video_overlay, heatmap_base, tracked_boxes, **{
                "frame_number": latest_metrics.frame_number,
                "total_frames": latest_metrics.total_frames,
                "video_time_s": latest_metrics.video_time_s,
                "total_duration_s": latest_metrics.total_duration_s,
                "visible_count": latest_metrics.visible_count,
                "estimated_count": latest_metrics.estimated_count,
                "peak_count": latest_metrics.peak_count,
                "fps": latest_metrics.fps,
                "latency_ms": latest_metrics.latency_ms,
                "density": latest_metrics.density,
                "cluster_count": latest_metrics.cluster_count,
                "largest_cluster": latest_metrics.largest_cluster,
                "risk_status": latest_metrics.risk_status,
                "risk_score": latest_metrics.risk_score,
            })
            await runtime.run_sync(combined_writer.write, combined_frame)

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
                # A live annotated-frame preview rides along with progress
                # (at the same throttled cadence as the broadcast itself, not
                # every processed frame) so the browser shows real boxes/
                # overlay WHILE the job is still running, not only once the
                # final video is ready. Never persisted to the job record --
                # only the numeric metrics are (see `update_video_progress`
                # above) -- this is a transient, WebSocket-only preview.
                progress_payload = latest_metrics.model_dump()
                try:
                    progress_payload["frame_jpeg"] = await runtime.run_sync(
                        encode_jpeg_data_url,
                        annotated,
                        settings.feed_frame_jpeg_quality,
                        settings.feed_frame_max_side,
                    )
                    progress_payload["heatmap_jpeg"] = await runtime.run_sync(
                        encode_jpeg_data_url,
                        combined_frame,
                        settings.feed_frame_jpeg_quality,
                        settings.feed_frame_max_side,
                    )
                except Exception:
                    logger.exception("Failed to encode live preview frame for job %s", job_id)
                await connection_manager.broadcast(
                    channel,
                    WSEnvelope(type=WSEventType.JOB_PROGRESS, job_id=job_id, data=progress_payload),
                )
                last_emit = now

        if frames_processed == 0 or latest_metrics is None:
            raise RuntimeError("No frames could be decoded from the uploaded video.")

        for writer, path, label in (
            (annotated_writer, annotated_path, "annotated"),
            (heatmap_writer, heatmap_path, "heatmap"),
            (combined_writer, combined_path, "combined"),
        ):
            if writer is not None:
                await runtime.run_sync(writer.release)
                if path.exists():
                    try:
                        await runtime.run_sync(ensure_browser_compatible_video, path)
                    except Exception:
                        logger.exception("Failed to transcode %s video for job %s", label, job_id)
        annotated_writer = None
        heatmap_writer = None
        combined_writer = None

        summary = VideoJobSummary(
            frames_processed=frames_processed,
            frames_detected=frames_detected,
            density_updates=density_updates,
            total_frames=total_frames,
            elapsed_s=round(time.time() - started, 3),
            peak_count=round(state.peak_count, 2),
            final_metrics=latest_metrics,
        )
        job_store.mark_completed(job_id, summary.model_dump())
        if annotated_path.exists():
            job_store.set_annotated_video_path(job_id, annotated_path)
        if heatmap_path.exists():
            job_store.set_heatmap_video_path(job_id, heatmap_path)
        if combined_path.exists():
            job_store.set_combined_video_path(job_id, combined_path)
        succeeded = True
        await connection_manager.broadcast(
            channel,
            WSEnvelope(type=WSEventType.JOB_COMPLETED, job_id=job_id, data=summary.model_dump()),
        )

    except Exception as exc:
        logger.exception("Video job %s failed", job_id)
        try:
            video_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove temp video for job %s", job_id, exc_info=True)
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
        for writer, label in (
            (annotated_writer, "annotated"),
            (heatmap_writer, "heatmap"),
            (combined_writer, "combined"),
        ):
            if writer is not None:
                try:
                    await runtime.run_sync(writer.release)
                except Exception:
                    logger.exception("Failed to release %s VideoWriter for job %s", label, job_id)
        try:
            video_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove temp video for job %s", job_id, exc_info=True)
        if not succeeded:
            # Nothing valid to serve if the job failed -- clean up any
            # partial output. On success the files are kept (see
            # `JobStore.set_*_video_path`) so `GET /api/jobs/{job_id}/video/*`
            # can actually serve them; they're removed later by the job's
            # normal TTL eviction instead.
            for path, label in ((annotated_path, "annotated"), (heatmap_path, "heatmap"), (combined_path, "combined")):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Failed to remove %s temp video for job %s", label, job_id, exc_info=True)
