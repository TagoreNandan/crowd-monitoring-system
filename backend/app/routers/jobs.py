"""Job endpoints:

    POST /api/jobs/image     -- synchronous: runs the existing pipeline and
                                 returns a completed job record.
    POST /api/jobs/video     -- accepts + stores the upload, creates an
                                 in-memory job record, schedules background
                                 frame-by-frame processing (see
                                 `jobs/video_runner.py`), and returns
                                 immediately with HTTP 202. Progress and the
                                 final result are published on
                                 `WS /ws/jobs/{job_id}` (`job.started` /
                                 `job.progress` / `job.completed` /
                                 `job.error`), not in this response.
    GET  /api/jobs/{job_id}  -- look up any job created above.

All inference goes through `PipelineRuntime.analyze()`, which is the only
place synchronous PyTorch inference is invoked -- routers never call the
pipeline directly on the event loop.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse

from crowd_monitoring.detection import PersonDetector

from ..dependencies import get_connection_manager, get_job_store, get_person_detector, get_pipeline_runtime
from ..errors import InvalidUploadError, JobNotFoundError, PipelineUnavailableError, VideoNotAvailableError
from ..jobs.enums import JobType
from ..jobs.store import JobStore
from ..jobs.video_runner import schedule_video_job
from ..pipeline_runtime import PipelineRuntime
from ..schemas.jobs import JobRecordResponse, frame_result_to_image_result, job_record_to_response
from ..settings import get_settings
from ..utils import decode_image_bytes, save_upload_to_tempfile
from ..vision_utils import build_image_visualizations, detect_persons_with_refinement_async
from ..ws.manager import ConnectionManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("/image", response_model=JobRecordResponse)
async def create_image_job(
    image: UploadFile = File(...),
    runtime: PipelineRuntime = Depends(get_pipeline_runtime),
    job_store: JobStore = Depends(get_job_store),
    person_detector: PersonDetector = Depends(get_person_detector),
) -> JobRecordResponse:
    settings = get_settings()
    max_bytes = int(settings.max_image_upload_mb * 1024 * 1024)

    contents = await image.read(max_bytes + 1)
    await image.close()
    if not contents:
        raise InvalidUploadError("Uploaded image is empty.")
    if len(contents) > max_bytes:
        raise InvalidUploadError(
            f"Uploaded image exceeds the maximum allowed size ({settings.max_image_upload_mb:.0f} MB)."
        )

    frame = decode_image_bytes(contents)

    job = job_store.create(job_type=JobType.IMAGE, source_description=image.filename or "upload")
    job_store.mark_running(job.job_id)
    try:
        # DM-Count (dense estimated_count/density/heatmap/localization/risk)
        # and the YOLOv8n person detector (visible_count/detections) are
        # separate models answering separate questions -- both run for
        # every image job, never conflated into one number.
        result = await runtime.analyze(frame)

        # Density-guided refinement (full-frame pass + selective re-detection
        # on DM-Count-identified dense regions, NMS-merged) -- a one-shot
        # image analysis has the latency budget for this; video/real-time
        # don't and use the plain full-frame detector instead.
        detector_started = time.time()
        detections, visible_count = await detect_persons_with_refinement_async(
            runtime.run_sync, person_detector, frame, result.inference.density_map, f"image job {job.job_id}"
        )
        detector_latency_ms = (time.time() - detector_started) * 1000.0 if visible_count is not None else None

        images = await runtime.run_sync(
            build_image_visualizations, frame, result, detections, person_detector.is_available
        )
    except Exception as exc:
        job_store.mark_failed(job.job_id, str(exc))
        raise PipelineUnavailableError(f"Image analysis failed: {exc}") from exc

    summary = frame_result_to_image_result(
        result,
        detections=detections,
        visible_count=visible_count,
        detector_latency_ms=detector_latency_ms,
        images=images,
    )
    job_store.mark_completed(job.job_id, summary.model_dump())
    return job_record_to_response(job_store.get(job.job_id))


@router.post("/video", response_model=JobRecordResponse, status_code=202)
async def create_video_job(
    video: UploadFile = File(...),
    runtime: PipelineRuntime = Depends(get_pipeline_runtime),
    job_store: JobStore = Depends(get_job_store),
    connection_manager: ConnectionManager = Depends(get_connection_manager),
    person_detector: PersonDetector = Depends(get_person_detector),
) -> JobRecordResponse:
    settings = get_settings()
    max_bytes = int(settings.max_video_upload_mb * 1024 * 1024)

    tmp_path = await save_upload_to_tempfile(video, max_bytes=max_bytes)

    job = job_store.create(
        job_type=JobType.VIDEO,
        source_description=video.filename or "upload",
        temp_path=tmp_path,
    )

    # Fire-and-forget: the response below is returned before this does any
    # work. `schedule_video_job` never runs inference or OpenCV I/O on this
    # coroutine -- it just schedules `run_video_job` on the event loop.
    schedule_video_job(
        job.job_id, tmp_path, runtime, job_store, connection_manager, person_detector,
        settings.density_analysis_hz, settings.video_detection_hz,
    )

    return job_record_to_response(job)


@router.get("/{job_id}", response_model=JobRecordResponse)
async def get_job(job_id: str, job_store: JobStore = Depends(get_job_store)) -> JobRecordResponse:
    record = job_store.get(job_id)
    if record is None:
        raise JobNotFoundError(job_id)
    return job_record_to_response(record)


@router.get("/{job_id}/video")
async def get_job_video(job_id: str, job_store: JobStore = Depends(get_job_store)) -> FileResponse:
    """Serve the real, annotated (person boxes + live stats overlay) output
    video for a completed video job -- H.264/yuv420p/faststart when ffmpeg
    was available at processing time (see `crowd_monitoring.video_encoding`), so it
    plays directly in a browser `<video>` element. Equivalent to
    `GET /{job_id}/video/detection` -- kept as the bare `/video` path for
    backward compatibility."""
    return _serve_job_video(job_id, job_store, view="detection")


@router.get("/{job_id}/video/{view}")
async def get_job_video_view(
    job_id: str, view: str, job_store: JobStore = Depends(get_job_store)
) -> FileResponse:
    """Serve one of the real output video views for a completed video job:

    - `detection` -- person boxes + stats panel (same as bare `/video`)
    - `heatmap`   -- real DM-Count density overlay + stats panel, no boxes
    - `combined`  -- density overlay + person boxes + stats panel together

    All three are produced in the SAME processing pass (see
    `jobs/video_runner.py`) -- `heatmap`/`combined` reuse the density map
    already computed for `detection`'s periodic DM-Count pass, never
    triggering extra inference."""
    if view not in ("detection", "heatmap", "combined"):
        raise VideoNotAvailableError(job_id, f"Unknown video view '{view}'. Use detection, heatmap, or combined.")
    return _serve_job_video(job_id, job_store, view=view)


def _serve_job_video(job_id: str, job_store: JobStore, view: str) -> FileResponse:
    record = job_store.get(job_id)
    if record is None:
        raise JobNotFoundError(job_id)
    path = {
        "detection": record.annotated_video_path,
        "heatmap": record.heatmap_video_path,
        "combined": record.combined_video_path,
    }[view]
    if path is None or not path.exists():
        raise VideoNotAvailableError(job_id, f"No '{view}' video is available for job '{job_id}' yet.")
    return FileResponse(path=path, media_type="video/mp4", filename=f"{job_id}_{view}.mp4")
