"""Real-Time Feed API Endpoints:

    POST   /api/feeds            -- Start a live feed session (sample video, camera, RTSP)
    GET    /api/feeds            -- List active/known feed sessions
    GET    /api/feeds/{feed_id}  -- Get status, metadata, latest metrics of a feed
    POST   /api/feeds/{feed_id}/stop -- Stop a running feed session

Architecture (why this file looks the way it does): a live feed cannot afford
to put the ~500ms+ DM-Count density pass on the same critical path as frame
delivery -- doing that is exactly what made an earlier version of this
feature visibly stall for half a second or more between frames. Instead this
uses `crowd_monitoring.streaming.realtime.RealtimeFeedSession`, which runs three
independent threads against one strict latest-frame-wins slot:

    producer (reads source)  ->  LatestSlot[Frame]  -+-> fast thread (detector + tracker, every available frame)
                                                       +-> density thread (DM-Count, paced to ~1/sec)

`on_fast_result` below fires every ~80-150ms (detector latency) with real
boxes/tracking and broadcasts the annotated video frame immediately --
completely independent of `on_density_result`, which fires roughly once a
second with a fresh heatmap/combined overlay and updated estimated-count/
density/risk. Neither callback blocks the other; neither ever processes a
backlog of stale frames (see `LatestSlot`/`RealtimeFeedSession` docstrings).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, List, Optional

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from crowd_monitoring.detection import PersonDetector
from crowd_monitoring.overlay import draw_density_heatmap_overlay, draw_person_boxes, render_video_overlay
from crowd_monitoring.pipeline import CrowdAnalysisPipeline
from crowd_monitoring.streaming import CameraSource, FileVideoSource, FrameSource, RTSPSource, RealtimeFeedSession
from crowd_monitoring.streaming.realtime import DensityResult, FastResult, RealtimeError
from crowd_monitoring.tracking import PersonTracker, TrackedPerson

from ..dependencies import get_connection_manager, get_person_detector, get_pipeline_runtime
from ..pipeline_runtime import PipelineRuntime
from ..schemas.ws import ErrorData, WSEnvelope, WSEventType
from ..settings import get_settings
from ..utils import save_upload_to_tempfile
from ..vision_utils import encode_jpeg_data_url
from ..ws.manager import ConnectionManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/feeds", tags=["feeds"])

# Global in-memory storage for active live feed sessions and their per-feed
# peak-count bookkeeping (peak is a running max over the feed's lifetime --
# not part of RealtimeFeedSession itself, which knows nothing about
# analytics semantics).
_active_feeds: dict[str, RealtimeFeedSession] = {}
_feed_peaks: dict[str, float] = {}


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
    # Fast-path (detector) latency/fps -- how often the live video/boxes are
    # actually updating. Distinct from `density_latency_ms`, which is
    # DM-Count's own (much slower, ~1/sec) per-update cost.
    latency_ms: float
    fps: Optional[float] = None
    density_latency_ms: Optional[float] = None
    frames_dropped: int = 0
    visible_count: Optional[int] = None
    frame_jpeg: Optional[str] = None
    heatmap_jpeg: Optional[str] = None
    combined_jpeg: Optional[str] = None


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


def _build_feed_response(session: RealtimeFeedSession, source_type: str) -> FeedSessionResponse:
    meta = session.metadata
    latest_metrics = None
    fast = session.latest_fast_result
    density = session.latest_density_result
    if fast is not None or density is not None:
        payload: dict[str, Any] = {
            "frame_number": fast.frame.frame_number if fast else (density.frame.frame_number if density else 0),
            "timestamp_s": round((fast.frame.timestamp_s if fast else density.frame.timestamp_s), 2) if (fast or density) else 0.0,
            "estimated_count": round(float(density.result.inference.count), 2) if density else 0.0,
            "peak_count": round(_feed_peaks.get(session.session_id, 0.0), 2),
            "density": 0.0,
            "cluster_count": int(density.result.clusters.n_clusters) if density else 0,
            "largest_cluster": int(density.result.clusters.largest_cluster_size) if density else 0,
            "risk_status": str(density.result.risk.status) if density else "SAFE",
            "risk_score": round(float(density.result.risk.score), 2) if density else 0.0,
            "latency_ms": round(fast.latency_ms, 2) if fast else 0.0,
            "fps": round(1000.0 / fast.latency_ms, 2) if fast and fast.latency_ms > 0 else None,
            "density_latency_ms": round(density.latency_ms, 2) if density else None,
            "frames_dropped": max(0, session.frames_produced - session.frames_detected),
            "visible_count": fast.visible_count if fast else None,
        }
        if density is not None:
            dcomp = density.result.risk.components.get("relative_density_per_100px2", 0.0)
            payload["density"] = round(float(dcomp or 0.0), 3)
        latest_metrics = FeedMetricsResponse(**payload)

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


def _resolve_youtube_stream_url(url: str) -> str:
    """Extract direct media stream URL from a YouTube URL via yt-dlp."""
    try:
        import yt_dlp
        ydl_opts = {
            "format": "best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info:
                if "url" in info:
                    return info["url"]
                elif "entries" in info and len(info["entries"]) > 0:
                    return info["entries"][0].get("url", url)
    except Exception as e:
        logger.warning(f"Failed to resolve YouTube stream URL via yt-dlp: {e}")
    return url


def _resolve_source(source_type: str, source_uri: Optional[str]) -> FrameSource:
    if source_type == "camera":
        idx = int(source_uri) if source_uri and source_uri.isdigit() else 0
        return CameraSource(device_index=idx)

    # Automatically resolve YouTube URLs for RTSP/RTMP or direct links
    if source_uri and ("youtube.com" in source_uri or "youtu.be" in source_uri):
        resolved_stream = _resolve_youtube_stream_url(source_uri)
        return RTSPSource(url=resolved_stream)

    if source_type in ("rtsp", "rtmp", "youtube"):
        if not source_uri:
            raise HTTPException(status_code=400, detail="RTSP/RTMP stream URL is required")
        return RTSPSource(url=source_uri)

    if source_type in ("video", "sample"):
        sample_path = Path(source_uri) if source_uri else Path("examples/sample_crowd.mp4")
        if not sample_path.exists():
            default_sample = Path("examples/sample_crowd.mp4")
            if default_sample.exists():
                sample_path = default_sample
            else:
                candidates = list(Path(".").glob("**/*.mp4"))
                if candidates:
                    sample_path = candidates[0]
                else:
                    raise HTTPException(status_code=404, detail=f"Sample video path not found: {sample_path}")
        return FileVideoSource(sample_path)
    raise HTTPException(status_code=400, detail=f"Unsupported source_type '{source_type}'")


def _scale_tracked_persons(
    persons: List[TrackedPerson], target_w: int, target_h: int, orig_w: int, orig_h: int
) -> List[TrackedPerson]:
    if not persons or (target_w == orig_w and target_h == orig_h):
        return persons
    sx = target_w / float(orig_w)
    sy = target_h / float(orig_h)
    scaled: List[TrackedPerson] = []
    for p in persons:
        track_id = int(getattr(p, "track_id", 0) or 0)
        x1 = int(round(p.x1 * sx))
        y1 = int(round(p.y1 * sy))
        x2 = int(round(p.x2 * sx))
        y2 = int(round(p.y2 * sy))
        confidence = float(p.confidence)
        center_xy = ((x1 + x2) // 2, (y1 + y2) // 2)
        history = [(int(round(hx * sx)), int(round(hy * sy))) for (hx, hy) in getattr(p, "history", [])]
        scaled.append(
            TrackedPerson(
                track_id=track_id,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=confidence,
                center_xy=center_xy,
                history=history,
            )
        )
    return scaled


@router.post("", response_model=FeedSessionResponse, status_code=201)
async def create_feed(
    req: CreateFeedRequest,
    runtime: PipelineRuntime = Depends(get_pipeline_runtime),
    connection_manager: ConnectionManager = Depends(get_connection_manager),
    person_detector: PersonDetector = Depends(get_person_detector),
) -> FeedSessionResponse:
    feed_id = f"feed_{uuid.uuid4().hex[:8]}"
    source = _resolve_source(req.source_type, req.source_uri)

    settings = get_settings()
    _feed_peaks[feed_id] = 0.0

    # `on_fast_result`/`on_density_result` run on RealtimeFeedSession's own
    # background threads, never on the asyncio event loop -- capture the
    # real loop here (this coroutine IS running on it) so both callbacks can
    # hand a broadcast back to it with `run_coroutine_threadsafe`. Calling
    # `asyncio.get_running_loop()` from those background threads directly
    # would always raise (no running loop on that thread).
    loop = asyncio.get_running_loop()
    channel = connection_manager.feed_channel(feed_id)

    def broadcast(envelope: WSEnvelope) -> None:
        asyncio.run_coroutine_threadsafe(connection_manager.broadcast(channel, envelope), loop)

    def on_fast_result(fast: FastResult) -> None:
        try:
            max_side = settings.feed_frame_max_side
            frame_bgr = fast.frame.frame_bgr
            fh, fw = frame_bgr.shape[:2]

            if max_side and max(fh, fw) > max_side:
                scale = max_side / float(max(fh, fw))
                target_w, target_h = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
                disp_frame = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
                disp_detections = _scale_tracked_persons(fast.detections, target_w, target_h, fw, fh)
            else:
                disp_frame = frame_bgr
                disp_detections = fast.detections

            annotated = render_video_overlay(
                disp_frame,
                disp_detections,
                frame_number=fast.frame.frame_number,
                total_frames=None,
                video_time_s=fast.frame.timestamp_s,
                total_duration_s=None,
                visible_count=fast.visible_count,
                estimated_count=_latest_density_count(feed_id),
                peak_count=_feed_peaks.get(feed_id, 0.0),
                fps=round(1000.0 / fast.latency_ms, 2) if fast.latency_ms > 0 else None,
                latency_ms=fast.latency_ms,
                density=_latest_density_value(feed_id),
                cluster_count=_latest_density_clusters(feed_id),
                largest_cluster=_latest_density_largest_cluster(feed_id),
                risk_status=_latest_density_risk_status(feed_id),
                risk_score=_latest_density_risk_score(feed_id),
            )
            frame_jpeg = encode_jpeg_data_url(
                annotated, quality=settings.feed_frame_jpeg_quality, max_side=None
            )
            data = {
                "frame_number": fast.frame.frame_number,
                "timestamp_s": round(fast.frame.timestamp_s, 2),
                "visible_count": fast.visible_count,
                "latency_ms": round(fast.latency_ms, 2),
                "fps": round(1000.0 / fast.latency_ms, 2) if fast.latency_ms > 0 else None,
                "frame_jpeg": frame_jpeg,
            }
            broadcast(WSEnvelope(type=WSEventType.FEED_PROGRESS, job_id=feed_id, data=data))
        except Exception:
            logger.exception("Error building/broadcasting fast-path frame for feed %s", feed_id)

    def on_density_result(density: DensityResult) -> None:
        try:
            r = density.result
            count = round(float(r.inference.count), 2)
            peak = max(_feed_peaks.get(feed_id, 0.0), count)
            _feed_peaks[feed_id] = peak
            dcomp = float(r.risk.components.get("relative_density_per_100px2", 0.0) or 0.0)

            frame_bgr = density.frame.frame_bgr
            fh, fw = frame_bgr.shape[:2]
            max_side = settings.feed_frame_max_side
            density_map = r.inference.density_map
            active_session = _active_feeds.get(feed_id)
            fast_boxes = active_session.latest_fast_result.detections if (active_session and active_session.latest_fast_result) else []

            if max_side and max(fh, fw) > max_side:
                scale = max_side / float(max(fh, fw))
                target_w, target_h = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
                disp_frame = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
                disp_boxes = _scale_tracked_persons(fast_boxes, target_w, target_h, fw, fh)
            else:
                disp_frame = frame_bgr
                disp_boxes = fast_boxes

            heatmap_frame = draw_density_heatmap_overlay(disp_frame, density_map, alpha=0.70)
            combined_frame = draw_density_heatmap_overlay(disp_frame, density_map, alpha=0.45)
            combined_frame = draw_person_boxes(combined_frame, disp_boxes)

            heatmap_jpeg = encode_jpeg_data_url(
                heatmap_frame, quality=settings.feed_frame_jpeg_quality, max_side=None
            )
            combined_jpeg = encode_jpeg_data_url(
                combined_frame, quality=settings.feed_frame_jpeg_quality, max_side=None
            )

            data = {
                "estimated_count": count,
                "peak_count": round(peak, 2),
                "density": round(dcomp, 3),
                "cluster_count": int(r.clusters.n_clusters),
                "largest_cluster": int(r.clusters.largest_cluster_size),
                "risk_status": str(r.risk.status),
                "risk_score": round(float(r.risk.score), 2),
                "density_latency_ms": round(density.latency_ms, 2),
                "heatmap_jpeg": heatmap_jpeg,
                "combined_jpeg": combined_jpeg,
            }
            broadcast(WSEnvelope(type=WSEventType.FEED_PROGRESS, job_id=feed_id, data=data))
        except Exception:
            logger.exception("Error building/broadcasting density-path update for feed %s", feed_id)

    def on_error(err: RealtimeError) -> None:
        try:
            envelope = WSEnvelope(
                type=WSEventType.FEED_ERROR,
                job_id=feed_id,
                data=ErrorData(code=f"feed_{err.stage}_error", message=str(err.error)).model_dump(),
            )
            broadcast(envelope)
        except Exception:
            logger.exception("Error broadcasting WS feed error for %s", feed_id)

    # CPU-forced, reduced-resolution pipeline dedicated to real-time feeds
    # (see `PipelineRuntime._build_realtime_density_pipeline_sync`) -- keeps
    # MPS entirely free for the fast detector thread below, so DM-Count
    # never contends with (and therefore never stalls) live frame delivery.
    pipeline: CrowdAnalysisPipeline = await runtime.get_realtime_density_pipeline()
    session = RealtimeFeedSession(
        session_id=feed_id,
        source=source,
        pipeline=pipeline,
        person_detector=person_detector,
        tracker=PersonTracker(),
        on_fast_result=on_fast_result,
        on_density_result=on_density_result,
        on_error=on_error,
        density_hz=settings.density_analysis_hz,
        loop_source=req.loop,
    )

    try:
        session.start()
    except Exception as exc:
        _feed_peaks.pop(feed_id, None)
        raise HTTPException(status_code=500, detail=f"Failed to start feed: {exc}") from exc

    _active_feeds[feed_id] = session
    return _build_feed_response(session, req.source_type)


def _latest_density(feed_id: str) -> Optional[DensityResult]:
    session = _active_feeds.get(feed_id)
    return session.latest_density_result if session else None


def _latest_density_count(feed_id: str) -> float:
    d = _latest_density(feed_id)
    return round(float(d.result.inference.count), 2) if d else 0.0


def _latest_density_value(feed_id: str) -> float:
    d = _latest_density(feed_id)
    if not d:
        return 0.0
    return round(float(d.result.risk.components.get("relative_density_per_100px2", 0.0) or 0.0), 3)


def _latest_density_clusters(feed_id: str) -> int:
    d = _latest_density(feed_id)
    return int(d.result.clusters.n_clusters) if d else 0


def _latest_density_largest_cluster(feed_id: str) -> int:
    d = _latest_density(feed_id)
    return int(d.result.clusters.largest_cluster_size) if d else 0


def _latest_density_risk_status(feed_id: str) -> str:
    d = _latest_density(feed_id)
    return str(d.result.risk.status) if d else "SAFE"


def _latest_density_risk_score(feed_id: str) -> float:
    d = _latest_density(feed_id)
    return round(float(d.result.risk.score), 2) if d else 0.0


@router.post("/upload", response_model=FeedSessionResponse, status_code=201)
async def create_upload_feed(
    video: UploadFile = File(...),
    loop: bool = Form(True),
    runtime: PipelineRuntime = Depends(get_pipeline_runtime),
    connection_manager: ConnectionManager = Depends(get_connection_manager),
    person_detector: PersonDetector = Depends(get_person_detector),
) -> FeedSessionResponse:
    """Start a real-time feed session from an uploaded video file.

    Saves the upload to temporary storage, feeds it to `RealtimeFeedSession`
    (fast detector path + periodic DM-Count + latest-frame-wins), and unlinks
    the temp file when stopped.
    """
    settings = get_settings()
    max_bytes = int(settings.max_video_upload_mb * 1024 * 1024)
    tmp_path = await save_upload_to_tempfile(video, max_bytes=max_bytes)

    req = CreateFeedRequest(
        source_type="video",
        source_uri=str(tmp_path),
        loop=loop,
    )
    res = await create_feed(req, runtime, connection_manager, person_detector)
    session = _active_feeds.get(res.feed_id)
    if session is not None:
        setattr(session, "_temp_upload_path", tmp_path)
    return res


@router.get("", response_model=List[FeedSessionResponse])
async def list_feeds() -> List[FeedSessionResponse]:
    return [_build_feed_response(sess, "feed") for sess in list(_active_feeds.values())]


@router.get("/{feed_id}", response_model=FeedSessionResponse)
async def get_feed(feed_id: str) -> FeedSessionResponse:
    sess = _active_feeds.get(feed_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"Feed '{feed_id}' not found")
    return _build_feed_response(sess, "feed")


@router.post("/{feed_id}/stop", response_model=FeedSessionResponse)
async def stop_feed(
    feed_id: str, connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> FeedSessionResponse:
    sess = _active_feeds.get(feed_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"Feed '{feed_id}' not found")
    sess.stop()
    tmp_path = getattr(sess, "_temp_upload_path", None)
    if tmp_path is not None and isinstance(tmp_path, Path):
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    response = _build_feed_response(sess, "feed")
    await connection_manager.broadcast(
        connection_manager.feed_channel(feed_id),
        WSEnvelope(type=WSEventType.FEED_STATUS, job_id=feed_id, data={"feed_id": feed_id, "status": "stopped"}),
    )
    return response
