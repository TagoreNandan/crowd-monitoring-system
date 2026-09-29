"""Request/response schemas for the job endpoints, plus the two converters
that translate real `crowd_monitoring` pipeline output / internal `JobRecord`
objects into API-shaped Pydantic models. All values here are copied from an
actual `FrameResult` or `JobRecord` -- nothing is synthesized.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, Field

from ..jobs.enums import JobStatus, JobType

if TYPE_CHECKING:
    from crowd_monitoring.detection import PersonDetection
    from crowd_monitoring.pipeline import FrameResult

    from ..jobs.store import JobRecord


class DetectionBox(BaseModel):
    """One real person bounding box from the YOLOv8n detector -- a separate
    signal from DM-Count's dense `count`, never derived from it."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class ImageVisualizations(BaseModel):
    """Separate visual outputs for the image page, each a
    `data:image/jpeg;base64,...` URL a browser can render directly. `None`
    for any view that couldn't be produced (e.g. `detection` when the person
    detector is unavailable) -- never a placeholder image standing in for a
    missing one.
    """

    detection: Optional[str] = None
    density: Optional[str] = None
    heatmap: Optional[str] = None
    localization: Optional[str] = None
    combined: Optional[str] = None


class ClusterSummary(BaseModel):
    n_clusters: int
    n_noise: int
    largest_cluster_size: int
    average_cluster_size: float


class RiskSummary(BaseModel):
    status: str
    score: float
    explanation: str
    components: dict[str, Optional[float]]


class ImageAnalysisResult(BaseModel):
    """The real output of one `CrowdAnalysisPipeline.analyze()` call plus the
    separate YOLOv8n person detector, re-shaped into a typed API response.

    `count` is DM-Count's dense, occlusion-robust estimate ("Estimated
    Crowd"); `visible_count`/`detections` are the person detector's real,
    individually-resolved bounding boxes ("Visible People"). These are
    deliberately never conflated -- see the top-level README and
    `crowd_monitoring.detection` for why a dense/occluded crowd can validly show
    `visible_count << count`.
    """

    count: float
    visible_count: Optional[int] = None
    detections: List[DetectionBox] = Field(default_factory=list)
    model_name: str
    model_weights: str
    device: str
    latency_ms: float
    detector_latency_ms: Optional[float] = None
    n_points: int
    clusters: ClusterSummary
    risk: RiskSummary
    images: Optional[ImageVisualizations] = None
    s3_urls: Optional[dict[str, str]] = None
    warnings: list[str] = Field(default_factory=list)


class VideoFrameMetrics(BaseModel):
    """Compact per-frame metrics for one processed frame of a video job --
    this is the whole `job.progress` payload, and also the `final_metrics`
    of a completed video job's summary. Deliberately excludes the density
    map / heatmap / raw points: those are large and never leave the
    process in this phase."""

    frame_number: int
    total_frames: Optional[int] = None
    # Wall-clock time since the job started processing -- for progress/ETA
    # display only. This is NOT the video's own timestamp (a slow processing
    # run can easily exceed the video's own duration) -- see `video_time_s`
    # for that. Keeping these as two distinctly-named fields (rather than
    # one `elapsed_s` whose meaning changed depending on the caller) is the
    # fix for a real bug where the burned-in video overlay and the UI/API
    # showed different "Time" values for the same frame.
    elapsed_s: float
    # The video's own playback position for this frame (frame_number /
    # source_fps), and the video's total duration (total_frames /
    # source_fps) if known. This is what both the burned-in overlay's
    # "Time: X / Y" line AND the UI's "Time" display must use -- always
    # read from this one field, never independently recomputed, so the two
    # can never drift apart.
    video_time_s: float = 0.0
    total_duration_s: Optional[float] = None
    # Two deliberately separate counts (never conflated): `visible_count` is
    # the YOLOv8n person detector's real bounding-box count for this
    # frame (`None` if the detector is unavailable, never a fake 0);
    # `estimated_count` is DM-Count's dense, occlusion-robust estimate --
    # the same value the image-job pipeline already produces.
    visible_count: Optional[int] = None
    estimated_count: float
    peak_count: float
    density: float
    cluster_count: int
    largest_cluster: int
    risk_status: str
    risk_score: float
    latency_ms: float
    fps: Optional[float] = None


class VideoJobSummary(BaseModel):
    """The real output of a completed video job -- aggregated frame-loop
    bookkeeping plus the last frame's metrics, nothing synthesized."""

    frames_processed: int
    # Honest cadence reporting: `frames_processed` frames were decoded and
    # written to the output video, but the detector and DM-Count only
    # actually RAN on `frames_detected`/`density_updates` of them (see
    # `jobs/video_runner.py` module docstring) -- frames in between reuse
    # the last real result rather than being independently analyzed.
    frames_detected: int = 0
    density_updates: int = 0
    total_frames: Optional[int] = None
    elapsed_s: float
    peak_count: float
    final_metrics: VideoFrameMetrics


class JobRecordResponse(BaseModel):
    job_id: str
    job_type: JobType
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0
    source_description: str = ""
    message: Optional[str] = None
    result: Optional[ImageAnalysisResult] = None
    video_result: Optional[VideoJobSummary] = None
    current_frame: int = 0
    total_frames: Optional[int] = None
    peak_count: float = 0.0
    latest_metrics: Optional[VideoFrameMetrics] = None
    error: Optional[str] = None


def frame_result_to_image_result(
    result: "FrameResult",
    detections: Optional[list["PersonDetection"]] = None,
    visible_count: Optional[int] = None,
    detector_latency_ms: Optional[float] = None,
    images: Optional[ImageVisualizations] = None,
    s3_urls: Optional[dict[str, str]] = None,
) -> ImageAnalysisResult:
    inf = result.inference
    return ImageAnalysisResult(
        count=round(inf.count, 2),
        visible_count=visible_count,
        detections=[DetectionBox(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2, confidence=round(d.confidence, 4)) for d in (detections or [])],
        model_name=inf.model_name,
        model_weights=inf.model_weights,
        device=inf.device,
        latency_ms=round(inf.latency_ms, 2),
        detector_latency_ms=round(detector_latency_ms, 2) if detector_latency_ms is not None else None,
        n_points=len(result.localization.points_xy),
        clusters=ClusterSummary(
            n_clusters=result.clusters.n_clusters,
            n_noise=result.clusters.n_noise,
            largest_cluster_size=result.clusters.largest_cluster_size,
            average_cluster_size=round(result.clusters.average_cluster_size, 2),
        ),
        risk=RiskSummary(
            status=result.risk.status,
            score=result.risk.score,
            explanation=result.risk.explanation,
            components=result.risk.components,
        ),
        images=images,
        s3_urls=s3_urls,
        warnings=list(inf.warnings),
    )


def job_record_to_response(record: "JobRecord") -> JobRecordResponse:
    image_result = None
    video_result = None
    if record.result is not None:
        if record.job_type == JobType.VIDEO:
            video_result = VideoJobSummary(**record.result)
        else:
            image_result = ImageAnalysisResult(**record.result)
    latest_metrics = VideoFrameMetrics(**record.latest_metrics) if record.latest_metrics is not None else None
    return JobRecordResponse(
        job_id=record.job_id,
        job_type=record.job_type,
        status=record.status,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        progress=record.progress,
        source_description=record.source_description,
        message=record.message,
        result=image_result,
        video_result=video_result,
        current_frame=record.current_frame,
        total_frames=record.total_frames,
        peak_count=record.peak_count,
        latest_metrics=latest_metrics,
        error=record.error,
    )
