"""Request/response schemas for the job endpoints, plus the two converters
that translate real `crowdmon` pipeline output / internal `JobRecord`
objects into API-shaped Pydantic models. All values here are copied from an
actual `FrameResult` or `JobRecord` -- nothing is synthesized.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

from ..jobs.enums import JobStatus, JobType

if TYPE_CHECKING:
    from crowdmon.pipeline import FrameResult

    from ..jobs.store import JobRecord


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
    """The real output of one `CrowdAnalysisPipeline.analyze()` call,
    re-shaped into a typed API response. `count` is DM-Count's dense,
    occlusion-robust estimate -- there is no separate visible-person
    detector in this phase, so no "visible count" field exists yet."""

    count: float
    model_name: str
    model_weights: str
    device: str
    latency_ms: float
    n_points: int
    clusters: ClusterSummary
    risk: RiskSummary
    warnings: list[str] = Field(default_factory=list)


class VideoFrameMetrics(BaseModel):
    """Compact per-frame metrics for one processed frame of a video job --
    this is the whole `job.progress` payload, and also the `final_metrics`
    of a completed video job's summary. Deliberately excludes the density
    map / heatmap / raw points: those are large and never leave the
    process in this phase."""

    frame_number: int
    total_frames: Optional[int] = None
    elapsed_s: float
    # Two deliberately separate counts (never conflated): `visible_count` is
    # the MobileNet-SSD person detector's real bounding-box count for this
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


def frame_result_to_image_result(result: "FrameResult") -> ImageAnalysisResult:
    inf = result.inference
    return ImageAnalysisResult(
        count=round(inf.count, 2),
        model_name=inf.model_name,
        model_weights=inf.model_weights,
        device=inf.device,
        latency_ms=round(inf.latency_ms, 2),
        n_points=int(len(result.localization.points_xy)),
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
