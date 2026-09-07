"""Real-Time Feed API Endpoints:

    POST   /api/feeds            -- Start a live feed session (sample video, camera, RTSP)
    GET    /api/feeds            -- List active/known feed sessions
    GET    /api/feeds/{feed_id}  -- Get status, metadata, latest metrics of a feed
    POST   /api/feeds/{feed_id}/stop -- Stop a running feed session
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional
from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, HTTPException, Query

from crowdmon.streaming import CameraSource, FileVideoSource, FrameSource, RTSPSource, StreamSession
from crowdmon.pipeline import CrowdAnalysisPipeline

from ..dependencies import get_connection_manager, get_pipeline_runtime
from ..pipeline_runtime import PipelineRuntime
from ..ws.manager import ConnectionManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/feeds", tags=["feeds"])

# Global in-memory storage for active live feed sessions
_active_feeds: dict[str, StreamSession] = {}


class CreateFeedRequest(BaseModel):
    source_type: str = Field("sample", description="Feed type: 'sample', 'video', 'camera', or 'rtsp'")
    source_uri: Optional[str] = Field(None, description="File path, RTSP/RTMP URL, or camera index")
    loop: bool = Field(True, description="Whether to loop video/sample source continuously")


class FeedMetricsResponse(BaseModel):
    frame_number: int
    timestamp_s: float
    estimated_count: float
    peak_count: float
    density: float
    cluster_count: int
    largest_cluster: int
    risk_status: str
    risk_score: float
    latency_ms: float
    fps: Optional[float] = None
    frames_dropped: int = 0


class FeedSessionResponse(BaseModel):
    feed_id: str
    source_type: str
    status: str
    description: str
    is_live: bool
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    frames_produced: int = 0
    error_count: int = 0
    started_at: Optional[float] = None
    latest_metrics: Optional[FeedMetricsResponse] = None


def _build_feed_response(session: StreamSession, source_type: str) -> FeedSessionResponse:
    meta = session.metadata
    latest_metrics = None
    if session.latest_result is not None:
        wr = session.latest_result
        r = wr.result
        density = float(r.risk.components.get("relative_density_per_100px2", 0.0) or 0.0)
        lat = float(r.inference.latency_ms)
        latest_metrics = FeedMetricsResponse(
            frame_number=wr.frame.frame_number,
            timestamp_s=round(wr.frame.timestamp_s, 2),
            estimated_count=round(float(r.inference.count), 2),
            peak_count=round(float(r.inference.count), 2),
            density=round(density, 3),
            cluster_count=int(r.clusters.n_clusters),
            largest_cluster=int(r.clusters.largest_cluster_size),
            risk_status=str(r.risk.status),
            risk_score=round(float(r.risk.score), 2),
            latency_ms=round(lat, 2),
            fps=round(1000.0 / lat, 2) if lat > 0 else None,
            frames_dropped=wr.frames_dropped,
        )

    return FeedSessionResponse(
        feed_id=session.session_id,
        source_type=source_type,
        status=session.status,
        description=meta.description if meta else "unknown",
        is_live=meta.is_live if meta else False,
        width=meta.width if meta else None,
        height=meta.height if meta else None,
        fps=meta.fps if meta else None,
        frames_produced=session.frames_produced,
        error_count=session.error_count,
        started_at=session.started_at,
        latest_metrics=latest_metrics,
    )


@router.post("", response_model=FeedSessionResponse, status_code=201)
async def create_feed(
    req: CreateFeedRequest,
    runtime: PipelineRuntime = Depends(get_pipeline_runtime),
    connection_manager: ConnectionManager = Depends(get_connection_manager),
) -> FeedSessionResponse:
    import uuid

    feed_id = f"feed_{uuid.uuid4().hex[:8]}"

    source: FrameSource
    if req.source_type == "camera":
        idx = int(req.source_uri) if req.source_uri and req.source_uri.isdigit() else 0
        source = CameraSource(device_index=idx)
    elif req.source_type in ("rtsp", "rtmp"):
        if not req.source_uri:
            raise HTTPException(status_code=400, detail="RTSP/RTMP source_uri is required")
        source = RTSPSource(url=req.source_uri)
    elif req.source_type in ("video", "sample"):
        sample_path = Path(req.source_uri) if req.source_uri else Path("examples/sample_crowd.mp4")
        if not sample_path.exists():
            # Search fallback sample files
            candidates = list(Path(".").glob("**/*.mp4"))
            if candidates:
                sample_path = candidates[0]
            else:
                raise HTTPException(status_code=404, detail=f"Sample video path not found: {sample_path}")
        source = FileVideoSource(sample_path)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported source_type '{req.source_type}'")

    from ..schemas.ws import WSEnvelope, WSEventType

    def on_feed_result(wr):
        try:
            r = wr.result
            density = float(r.risk.components.get("relative_density_per_100px2", 0.0) or 0.0)
            lat = float(r.inference.latency_ms)
            metrics = {
                "frame_number": wr.frame.frame_number,
                "timestamp_s": round(wr.frame.timestamp_s, 2),
                "estimated_count": round(float(r.inference.count), 2),
                "density": round(density, 3),
                "cluster_count": int(r.clusters.n_clusters),
                "largest_cluster": int(r.clusters.largest_cluster_size),
                "risk_status": str(r.risk.status),
                "risk_score": round(float(r.risk.score), 2),
                "latency_ms": round(lat, 2),
                "fps": round(1000.0 / lat, 2) if lat > 0 else None,
                "frames_dropped": wr.frames_dropped,
            }
            envelope = WSEnvelope(
                type=WSEventType.FEED_PROGRESS,
                job_id=feed_id,
                data=metrics,
            )
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(connection_manager.broadcast(connection_manager.feed_channel(feed_id), envelope))
            except RuntimeError:
                pass
        except Exception:
            logger.exception("Error broadcasting WS feed metric for %s", feed_id)

    pipeline: CrowdAnalysisPipeline = await runtime.get_pipeline()
    session = StreamSession(
        session_id=feed_id,
        source=source,
        pipeline=pipeline,
        on_result=on_feed_result,
        buffer_maxsize=1,
        loop_source=req.loop,
    )

    try:
        session.start()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start feed: {exc}") from exc

    _active_feeds[feed_id] = session
    return _build_feed_response(session, req.source_type)


@router.get("", response_model=List[FeedSessionResponse])
async def list_feeds() -> List[FeedSessionResponse]:
    out = []
    for fid, sess in list(_active_feeds.items()):
        out.append(_build_feed_response(sess, "feed"))
    return out


@router.get("/{feed_id}", response_model=FeedSessionResponse)
async def get_feed(feed_id: str) -> FeedSessionResponse:
    sess = _active_feeds.get(feed_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"Feed '{feed_id}' not found")
    return _build_feed_response(sess, "feed")


@router.post("/{feed_id}/stop", response_model=FeedSessionResponse)
async def stop_feed(feed_id: str) -> FeedSessionResponse:
    sess = _active_feeds.get(feed_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"Feed '{feed_id}' not found")
    sess.stop()
    return _build_feed_response(sess, "feed")
