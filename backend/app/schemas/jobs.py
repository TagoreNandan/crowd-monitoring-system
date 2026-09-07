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
    result = ImageAnalysisResult(**record.result) if record.result is not None else None
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
        result=result,
        error=record.error,
    )
