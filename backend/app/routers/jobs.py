"""Job endpoints:

    POST /api/jobs/image     -- synchronous: runs the existing pipeline and
                                 returns a completed job record.
    POST /api/jobs/video     -- accepts + stores the upload, creates an
                                 in-memory job record, returns immediately.
                                 Frame-by-frame processing is NOT implemented
                                 in this phase (that's the Phase 2 background
                                 runner) -- see the job's `message` field.
    GET  /api/jobs/{job_id}  -- look up any job created above.

All inference goes through `PipelineRuntime.analyze()`, which is the only
place synchronous PyTorch inference is invoked -- routers never call the
pipeline directly on the event loop.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, UploadFile

from ..dependencies import get_job_store, get_pipeline_runtime
from ..errors import InvalidUploadError, JobNotFoundError, PipelineUnavailableError
from ..jobs.enums import JobType
from ..jobs.store import JobStore
from ..pipeline_runtime import PipelineRuntime
from ..schemas.jobs import JobRecordResponse, frame_result_to_image_result, job_record_to_response
from ..settings import get_settings
from ..utils import decode_image_bytes, save_upload_to_tempfile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("/image", response_model=JobRecordResponse)
async def create_image_job(
    image: UploadFile = File(...),
    runtime: PipelineRuntime = Depends(get_pipeline_runtime),
    job_store: JobStore = Depends(get_job_store),
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
        result = await runtime.analyze(frame)
    except Exception as exc:
        job_store.mark_failed(job.job_id, str(exc))
        raise PipelineUnavailableError(f"Image analysis failed: {exc}") from exc

    summary = frame_result_to_image_result(result)
    job_store.mark_completed(job.job_id, summary.model_dump())
    return job_record_to_response(job_store.get(job.job_id))


@router.post("/video", response_model=JobRecordResponse, status_code=202)
async def create_video_job(
    video: UploadFile = File(...),
    job_store: JobStore = Depends(get_job_store),
) -> JobRecordResponse:
    settings = get_settings()
    max_bytes = int(settings.max_video_upload_mb * 1024 * 1024)

    tmp_path = await save_upload_to_tempfile(video, max_bytes=max_bytes)

    job = job_store.create(
        job_type=JobType.VIDEO,
        source_description=video.filename or "upload",
        temp_path=tmp_path,
        message=(
            "Video received and stored in a temporary working file. "
            "Frame-by-frame processing is not implemented in this phase; "
            "this job record establishes the job/WebSocket contract for "
            "a later phase's background runner."
        ),
    )
    return job_record_to_response(job)


@router.get("/{job_id}", response_model=JobRecordResponse)
async def get_job(job_id: str, job_store: JobStore = Depends(get_job_store)) -> JobRecordResponse:
    record = job_store.get(job_id)
    if record is None:
        raise JobNotFoundError(job_id)
    return job_record_to_response(record)
