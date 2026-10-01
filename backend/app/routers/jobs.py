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

import base64
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response

from crowd_monitoring.detection import PersonDetector

from ..auth.dependencies import get_current_user, log_activity
from ..db.models import User
from ..db.session import get_db
from ..dependencies import get_connection_manager, get_job_store, get_person_detector, get_pipeline_runtime
from ..errors import InvalidUploadError, JobNotFoundError, PipelineUnavailableError, VideoNotAvailableError
from ..jobs.enums import JobType
from ..jobs.store import JobStore
from ..jobs.video_runner import schedule_video_job
from ..pipeline_runtime import PipelineRuntime
from ..schemas.jobs import JobRecordResponse, frame_result_to_image_result, job_record_to_response
from ..settings import get_settings
from .. import storage
from ..utils import decode_image_bytes, save_upload_to_tempfile
from ..vision_utils import build_image_visualizations, detect_persons_with_refinement_async
from ..ws.manager import ConnectionManager
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("/image", response_model=JobRecordResponse)
async def create_image_job(
    image: UploadFile = File(...),
    runtime: PipelineRuntime = Depends(get_pipeline_runtime),
    job_store: JobStore = Depends(get_job_store),
    person_detector: PersonDetector = Depends(get_person_detector),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
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
        s3_urls: dict[str, str] = {}
        if storage.is_s3_enabled():
            for view in ("detection", "density", "heatmap", "localization", "combined"):
                data_url = getattr(images, view)
                if data_url:
                    try:
                        _, b64data = data_url.split(",", 1)
                        img_bytes = base64.b64decode(b64data)
                        key = f"jobs/{job.job_id}/image/{view}.jpg"
                        await runtime.run_sync(storage.upload_bytes, img_bytes, key, "image/jpeg")
                        job_store.set_image_storage_key(job.job_id, view, key)
                        s3_urls[view] = storage.presigned_url(key)
                    except Exception:
                        logger.exception("Failed to upload %s image for job %s to S3", view, job.job_id)
    except Exception as exc:
        job_store.mark_failed(job.job_id, str(exc))
        log_activity(
            db,
            action="IMAGE_ANALYSIS_FAILED",
            category="INFERENCE",
            details=f"Analysis failed for {image.filename or 'upload'}: {exc}",
            severity="CRITICAL",
            user=user,
        )
        raise PipelineUnavailableError(f"Image analysis failed: {exc}") from exc

    summary = frame_result_to_image_result(
        result,
        detections=detections,
        visible_count=visible_count,
        detector_latency_ms=detector_latency_ms,
        images=images,
        s3_urls=s3_urls if s3_urls else None,
    )
    job_store.mark_completed(job.job_id, summary.model_dump())

    # Log successful analysis activity
    risk_level = result.risk.level if hasattr(result, "risk") and hasattr(result.risk, "level") else "UNKNOWN"
    count = round(result.count, 1) if hasattr(result, "count") else 0
    severity = "CRITICAL" if str(risk_level).upper() in ("CRITICAL", "WARNING") else "INFO"
    image_output_keys = job.image_storage_keys.copy()
    if not image_output_keys:
        image_output_keys = {
            view: f"/api/jobs/{job.job_id}/image/{view}"
            for view in ("detection", "density", "heatmap", "localization", "combined")
            if getattr(images, view)
        }
    log_activity(
        db,
        action="IMAGE_ANALYZED",
        category="INFERENCE",
        details=f"File: {image.filename or 'upload'} | Count: {count} | Risk: {risk_level}",
        severity=severity,
        user=user,
        job_id=job.job_id,
        job_type="IMAGE",
        output_keys=image_output_keys,
        storage_backend="s3" if job.image_storage_keys else "local",
    )

    record = job_store.get(job.job_id)
    if record is None:
        raise JobNotFoundError(job.job_id)
    return job_record_to_response(record)


@router.post("/video", response_model=JobRecordResponse, status_code=202)
async def create_video_job(
    video: UploadFile = File(...),
    runtime: PipelineRuntime = Depends(get_pipeline_runtime),
    job_store: JobStore = Depends(get_job_store),
    connection_manager: ConnectionManager = Depends(get_connection_manager),
    person_detector: PersonDetector = Depends(get_person_detector),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> JobRecordResponse:
    settings = get_settings()
    max_bytes = int(settings.max_video_upload_mb * 1024 * 1024)

    tmp_path = await save_upload_to_tempfile(video, max_bytes=max_bytes)

    job = job_store.create(
        job_type=JobType.VIDEO,
        source_description=video.filename or "upload",
        temp_path=tmp_path,
        user_id=user.id if user else None,
    )

    schedule_video_job(
        job.job_id, tmp_path, runtime, job_store, connection_manager, person_detector,
        settings.density_analysis_hz, settings.video_detection_hz,
    )

    log_activity(
        db,
        action="VIDEO_JOB_SUBMITTED",
        category="INFERENCE",
        details=f"Video queued: {video.filename or 'upload'} (Job ID: {job.job_id})",
        severity="INFO",
        user=user,
    )

    return job_record_to_response(job)


@router.get("/{job_id}", response_model=JobRecordResponse)
async def get_job(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
    user: User = Depends(get_current_user),
) -> JobRecordResponse:
    record = job_store.get(job_id)
    if record is None:
        raise JobNotFoundError(job_id)
    return job_record_to_response(record)


@router.get("/{job_id}/video")
async def get_job_video(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
    user: User = Depends(get_current_user),
) -> Response:
    """Serve the real, annotated (person boxes + live stats overlay) output
    video for a completed video job -- H.264/yuv420p/faststart when ffmpeg
    was available at processing time (see `crowd_monitoring.video_encoding`), so it
    plays directly in a browser `<video>` element. Equivalent to
    `GET /{job_id}/video/detection` -- kept as the bare `/video` path for
    backward compatibility."""
    return _serve_job_video(job_id, job_store, view="detection")


@router.get("/{job_id}/video/{view}")
async def get_job_video_view(
    job_id: str,
    view: str,
    job_store: JobStore = Depends(get_job_store),
    user: User = Depends(get_current_user),
) -> Response:
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


@router.get("/jobs/{job_id}/video/{view}/url")
async def get_job_video_url_info(
    job_id: str,
    view: str = "detection",
    job_store: JobStore = Depends(get_job_store),
    user: User = Depends(get_current_user),
):
    """Return direct URL metadata (mode, pre-signed URL or local route, expires_in, expires_at)
    for frontend video player auto-renewal and uninterrupted viewing sessions."""
    if view not in ("detection", "heatmap", "combined"):
        raise VideoNotAvailableError(job_id, f"Unknown video view '{view}'. Use detection, heatmap, or combined.")
    record = job_store.get(job_id)
    if record is None:
        raise JobNotFoundError(job_id)

    storage_key = record.video_storage_keys.get(view)
    if storage_key:
        info = storage.get_presigned_url_info(storage_key)
        return {
            "mode": "s3",
            "view": view,
            "url": info["url"],
            "expires_in": info["expires_in"],
            "expires_at": info["expires_at"],
            "storage_key": storage_key,
        }

    path = {
        "detection": record.annotated_video_path,
        "heatmap": record.heatmap_video_path,
        "combined": record.combined_video_path,
    }[view]
    if path is None or not path.exists():
        raise VideoNotAvailableError(job_id, f"No '{view}' video is available for job '{job_id}' yet.")

    return {
        "mode": "local",
        "view": view,
        "url": f"/api/jobs/{job_id}/video/{view}",
        "expires_in": None,
        "expires_at": None,
    }


def _serve_job_video(job_id: str, job_store: JobStore, view: str):
    record = job_store.get(job_id)
    if record is None:
        raise JobNotFoundError(job_id)

    # S3 mode: if this view was uploaded, redirect straight to a temporary
    # signed S3 URL instead of streaming it through FastAPI.
    # Set Cache-Control so the browser never caches the 307 redirect when tokens expire!
    storage_key = record.video_storage_keys.get(view)
    if storage_key:
        return RedirectResponse(
            url=storage.presigned_url(storage_key),
            status_code=307,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    # Local mode (or S3 upload hasn't finished / failed): serve local file
    # exactly as before.
    path = {
        "detection": record.annotated_video_path,
        "heatmap": record.heatmap_video_path,
        "combined": record.combined_video_path,
    }[view]
    if path is None or not path.exists():
        raise VideoNotAvailableError(job_id, f"No '{view}' video is available for job '{job_id}' yet.")
    return FileResponse(path=path, media_type="video/mp4", filename=f"{job_id}_{view}.mp4")


@router.get("/jobs/{job_id}/image/{view}")
async def get_job_image(
    job_id: str,
    view: str = "detection",
    job_store: JobStore = Depends(get_job_store),
    user: User = Depends(get_current_user),
):
    """Serve or redirect to image outputs (detection, density, heatmap, localization, combined).
    If uploaded to S3, returns a 307 redirect to the temporary pre-signed URL."""
    if view not in ("detection", "density", "heatmap", "localization", "combined"):
        raise HTTPException(status_code=400, detail=f"Unknown image view '{view}'")
    record = job_store.get(job_id)
    if record is None:
        raise JobNotFoundError(job_id)

    storage_key = record.image_storage_keys.get(view)
    if storage_key:
        return RedirectResponse(
            url=storage.presigned_url(storage_key),
            status_code=307,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    if record.result and "images" in record.result and record.result["images"]:
        data_url = record.result["images"].get(view)
        if data_url and "," in data_url:
            _, b64data = data_url.split(",", 1)
            img_bytes = base64.b64decode(b64data)
            return Response(content=img_bytes, media_type="image/jpeg")

    raise HTTPException(status_code=404, detail=f"Image view '{view}' not found for job '{job_id}'")


@router.get("/jobs/{job_id}/image/{view}/url")
async def get_job_image_url_info(
    job_id: str,
    view: str = "detection",
    job_store: JobStore = Depends(get_job_store),
    user: User = Depends(get_current_user),
):
    """Return direct URL metadata for image outputs (mode, pre-signed URL, expires_in, expires_at)."""
    if view not in ("detection", "density", "heatmap", "localization", "combined"):
        raise HTTPException(status_code=400, detail=f"Unknown image view '{view}'")
    record = job_store.get(job_id)
    if record is None:
        raise JobNotFoundError(job_id)

    storage_key = record.image_storage_keys.get(view)
    if storage_key:
        info = storage.get_presigned_url_info(storage_key)
        return {
            "mode": "s3",
            "view": view,
            "url": info["url"],
            "expires_in": info["expires_in"],
            "expires_at": info["expires_at"],
            "storage_key": storage_key,
        }

    return {
        "mode": "local",
        "view": view,
        "url": f"/api/jobs/{job_id}/image/{view}",
        "expires_in": None,
        "expires_at": None,
    }
