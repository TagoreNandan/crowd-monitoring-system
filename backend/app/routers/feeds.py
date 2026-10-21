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
import threading
import time
import uuid
from pathlib import Path
from typing import Any, List, Optional

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from crowd_monitoring.detection import PersonDetector
from crowd_monitoring.overlay import draw_density_heatmap_overlay, draw_person_boxes, render_video_overlay
from crowd_monitoring.pipeline import CrowdAnalysisPipeline
from crowd_monitoring.streaming import (
    CameraSource,
    FailoverFrameSource,
    FileVideoSource,
    FrameSource,
    RTSPSource,
    RealtimeFeedSession,
    is_live_stream_url,
)
from crowd_monitoring.streaming.failover import MODE_AUTO, MODE_BACKUP
from crowd_monitoring.streaming.realtime import DensityResult, FastResult, RealtimeError
from crowd_monitoring.tracking import PersonTracker, TrackedPerson

from ..dependencies import get_connection_manager, get_person_detector, get_pipeline_runtime
from ..auth.dependencies import get_current_user, require_admin
from ..db.models import User
from .. import storage
from ..live_recording import LiveFeedRecorder
from ..pipeline_runtime import PipelineRuntime
from ..schemas.ws import ErrorData, WSEnvelope, WSEventType
from ..settings import get_settings
from ..paths import DEFAULT_SAMPLE_VIDEO, EXAMPLES_DIR, LIVE_RECORDINGS_DIR, find_local_file, is_within
from ..utils import parse_s3_source, redact_presigned_url, save_upload_to_tempfile
from ..vision_utils import encode_jpeg_data_url
from ..ws.manager import ConnectionManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/feeds", tags=["feeds"])

# Global in-memory storage for active live feed sessions and their per-feed
# peak-count bookkeeping (peak is a running max over the feed's lifetime --
# not part of RealtimeFeedSession itself, which knows nothing about
# analytics semantics). Guarded by _feeds_lock for thread-safety.
_feeds_lock = threading.Lock()
_active_feeds: dict[str, RealtimeFeedSession] = {}
_feed_peaks: dict[str, float] = {}
_feed_latest_frames: dict[str, dict[str, str]] = {}
_feed_recorders: dict[str, LiveFeedRecorder] = {}
_feed_device_ids: dict[str, str] = {}
_feed_device_names: dict[str, str] = {}
_starting_devices: dict[str, str] = {}

# Per-drone backup archives for live flights: when drone_N's stream drops,
# FailoverFrameSource loops this clip until the link recovers. Keyed by fleet
# device_id (or feed_id), case-insensitive. Bare keys resolve against
# S3_BUCKET_NAME; full s3://bucket/key URIs and local paths also work.
# A request's explicit `fallback_uri` wins; unmapped feeds fall back to
# settings.feed_failover_source.
DRONE_FALLBACK_VIDEOS: dict[str, str] = {
    "drone_1": "failsafe/backup_1.mp4",
    "drone_2": "failsafe/backup_2.mp4",
    "drone_3": "failsafe/backup_3.mp4",
    "drone_4": "failsafe/backup_4.mp4",
    "drone_5": "failsafe/backup_5.mp4",
    "drone_6": "failsafe/backup_6.mp4",
}


def drone_fallback_uri(*ids: Optional[str]) -> Optional[str]:
    """First DRONE_FALLBACK_VIDEOS entry matching any of `ids`, else None."""
    for ident in ids:
        if ident and ident.strip().lower() in DRONE_FALLBACK_VIDEOS:
            return DRONE_FALLBACK_VIDEOS[ident.strip().lower()]
    return None


def presigned_s3_video(uri: str, expires_in: int = 86400) -> tuple[str, str]:
    """Sign an S3 video (s3://bucket/key or bare key) on the fly for OpenCV.
    Returns (signed_url, redacted s3:// description safe to log/display)."""
    bucket, key = parse_s3_source(uri)
    url = storage.presigned_url(
        key=key, bucket=bucket, expires_in=expires_in, response_content_type="video/mp4"
    )
    return url, f"s3://{bucket}/{key}"


def get_active_feeds_snapshot() -> tuple[dict[str, RealtimeFeedSession], dict[str, float], dict[str, dict[str, str]]]:
    """Return a thread-safe snapshot of all active feeds and their state."""
    with _feeds_lock:
        return (
            dict(_active_feeds),
            dict(_feed_peaks),
            {k: dict(v) for k, v in _feed_latest_frames.items()},
        )


class CreateFeedRequest(BaseModel):
    source_type: str = Field("sample", description="Feed type: 'sample', 'video', 'camera', or 'rtsp'")
    source_uri: Optional[str] = Field(None, description="File path, RTSP/RTMP URL, or camera index")
    loop: bool = Field(True, description="Whether to loop video/sample source continuously")
    device_id: Optional[str] = Field(None, description="Optional associated fleet device ID")
    device_name: Optional[str] = Field(None, description="Optional associated fleet device name")
    fallback_uri: Optional[str] = Field(
        None,
        description=(
            "Backup video (S3 key, s3:// URI, or local path) played automatically if a live "
            "RTSP/RTMP stream drops. Defaults to the server's feed_failover_source; '' disables."
        ),
    )


class FeedSourceModeRequest(BaseModel):
    mode: str = Field(..., pattern="^(auto|backup)$", description="'auto' (failover on drop) or 'backup' (force backup)")


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
    is_recording: bool = False
    latest_metrics: Optional[FeedMetricsResponse] = None
    device_id: Optional[str] = None
    device_name: Optional[str] = None
    # "LIVE" | "BACKUP_ACTIVE" | "SIGNAL_LOST" (see crowd_monitoring.streaming.failover)
    source_status: str = "LIVE"
    failover_enabled: bool = False
    source_mode: Optional[str] = None


def _build_feed_response(session: RealtimeFeedSession, source_type: str) -> FeedSessionResponse:
    meta = session.metadata
    latest_metrics = None
    fast = session.latest_fast_result
    density = session.latest_density_result
    with _feeds_lock:
        peak_count = round(_feed_peaks.get(session.session_id, 0.0), 2)
        is_recording = session.session_id in _feed_recorders
        device_id = _feed_device_ids.get(session.session_id)
        device_name = _feed_device_names.get(session.session_id)
    if fast is not None or density is not None:
        active_frame = fast.frame if fast is not None else (density.frame if density is not None else None)
        frame_number = active_frame.frame_number if active_frame is not None else 0
        timestamp_s = round(active_frame.timestamp_s, 2) if active_frame is not None else 0.0
        payload: dict[str, Any] = {
            "frame_number": frame_number,
            "timestamp_s": timestamp_s,
            "estimated_count": round(float(density.result.inference.count), 2) if density else 0.0,
            "peak_count": peak_count,
            "density": 0.0,
            "cluster_count": density.result.clusters.n_clusters if density else 0,
            "largest_cluster": density.result.clusters.largest_cluster_size if density else 0,
            "risk_status": density.result.risk.status if density else "SAFE",
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
        is_recording=is_recording,
        latest_metrics=latest_metrics,
        device_id=device_id,
        device_name=device_name,
        source_status=session.source_status,
        failover_enabled=isinstance(session.source, FailoverFrameSource),
        source_mode=session.source.mode if isinstance(session.source, FailoverFrameSource) else None,
    )


def _resolve_youtube_stream_url(url: str) -> str:
    """Extract direct media stream URL from a YouTube URL via yt-dlp."""
    try:
        import yt_dlp
        ydl_opts = {
            "format": "bestvideo[height<=1080]/bestvideo/best",
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info and isinstance(info, dict):
                stream_url = info.get("url")
                if stream_url:
                    return stream_url
                entries = info.get("entries")
                if entries:
                    for entry in entries:
                        if isinstance(entry, dict) and entry.get("url"):
                            return str(entry["url"])
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Failed to resolve YouTube stream URL via yt-dlp: {e}")
        msg = str(e)
        if "This video is unavailable" in msg:
            raise HTTPException(
                status_code=400,
                detail="YouTube video is unavailable. Please verify the URL and casing (YouTube video IDs are case-sensitive).",
            ) from e
        err_line = msg.splitlines()[-1] if msg else "unknown error"
        raise HTTPException(
            status_code=400,
            detail=f"Unable to extract video stream from YouTube URL: {err_line}",
        ) from e
    raise HTTPException(status_code=400, detail="Could not extract video stream from the provided YouTube URL.")


def _resolve_source(source_type: str, source_uri: Optional[str]) -> FrameSource:
    if source_type == "camera":
        idx = int(source_uri) if source_uri and source_uri.isdigit() else 0
        return CameraSource(device_index=idx)

    # S3 Source Resolution (explicit source_type == 's3', s3:// URI, or object key like failsafe/drone_standby.mp4)
    is_s3_type = (source_type or "").lower() == "s3"
    is_s3_uri = bool(source_uri and source_uri.strip().startswith("s3://"))
    is_s3_key = bool(
        source_uri
        and "://" not in source_uri  # rtsp://.../live.mp4 etc. are URLs, not S3 keys
        and ("failsafe/" in source_uri or source_uri.endswith(".mp4"))
        and find_local_file(source_uri) is None
    )

    if is_s3_type or is_s3_uri or (source_type in ("video", "sample", "rtsp") and is_s3_key):
        s3_target = source_uri if (source_uri and source_uri.strip()) else "failsafe/drone_standby.mp4"
        try:
            if not storage.is_s3_enabled():
                raise HTTPException(
                    status_code=533,
                    detail="S3 storage backend is not enabled or boto3 is unavailable."
                )
            presigned, clean_descr = presigned_s3_video(s3_target)
            return RTSPSource(url=presigned, display_name=clean_descr)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to resolve S3 feed source: {type(e).__name__} - {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Unable to resolve S3 video source: {e}"
            ) from e

    # Automatically resolve YouTube URLs for RTSP/RTMP or direct links
    if source_uri and ("youtube.com" in source_uri or "youtu.be" in source_uri):
        resolved_stream = _resolve_youtube_stream_url(source_uri)
        return RTSPSource(url=resolved_stream)

    if source_type in ("rtsp", "rtmp", "youtube"):
        if not source_uri:
            raise HTTPException(status_code=400, detail="RTSP/RTMP stream URL is required")
        return RTSPSource(url=source_uri, max_backoff_s=get_settings().feed_reconnect_max_backoff_s)

    if source_type in ("video", "sample"):
        sample_path = find_local_file(source_uri) or Path(source_uri or DEFAULT_SAMPLE_VIDEO)
        if not sample_path.exists():
            if DEFAULT_SAMPLE_VIDEO.exists():
                sample_path = DEFAULT_SAMPLE_VIDEO
            else:
                candidates = sorted(EXAMPLES_DIR.glob("*.mp4"))
                if candidates:
                    sample_path = candidates[0]
                elif storage.is_s3_enabled():
                    # Fallback to S3 drone standby video if local sample video is missing
                    presigned, clean_descr = presigned_s3_video("failsafe/drone_standby.mp4")
                    return RTSPSource(url=presigned, display_name=clean_descr)
                else:
                    raise HTTPException(status_code=404, detail=f"Sample video path not found: {sample_path}")

        # Any absolute path on the host is allowed (feed creation is admin-only).
        return FileVideoSource(sample_path)
    raise HTTPException(status_code=400, detail=f"Unsupported source_type '{source_type}'")


def _make_backup_factory(fallback_uri: str):
    """Return a zero-arg factory resolving the backup video, or None if it
    can't be resolved here (S3 disabled and no such local file). Resolved
    afresh on every failover so presigned S3 URLs never go stale, and set to
    loop (seek to frame 0 at EOF) so the archive never stops mid-outage."""
    uri = fallback_uri.strip()
    if not uri:
        return None
    local = find_local_file(uri)
    if local is None and not storage.is_s3_enabled():
        logger.warning("Failover backup '%s' not found locally and S3 is disabled; failover off", uri)
        return None
    source_type = "s3" if local is None else "video"

    def factory() -> FrameSource:
        backup = _resolve_source(source_type, uri)
        if isinstance(backup, (FileVideoSource, RTSPSource)):
            backup.loop = True
        return backup

    return factory


def _wrap_with_failover(
    source: FrameSource, fallback_uri: Optional[str], on_status_change
) -> FrameSource:
    """Wrap live network streams (RTSP/RTMP/SRT...) in FailoverFrameSource.
    Files, cameras and finite HTTP media are returned unchanged."""
    settings = get_settings()
    if not settings.feed_failover_enabled or not isinstance(source, RTSPSource):
        return source
    if not is_live_stream_url(source.url):
        return source
    factory = _make_backup_factory(settings.feed_failover_source if fallback_uri is None else fallback_uri)
    if factory is None:
        return source
    return FailoverFrameSource(
        primary=source,
        backup_factory=factory,
        failover_after_s=settings.feed_failover_after_s,
        recover_after_s=settings.feed_failover_recover_after_s,
        primary_retry_max_s=settings.feed_reconnect_max_backoff_s,
        on_status_change=on_status_change,
    )


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
        x1 = round(p.x1 * sx)
        y1 = round(p.y1 * sy)
        x2 = round(p.x2 * sx)
        y2 = round(p.y2 * sy)
        confidence = float(p.confidence)
        center_xy = ((x1 + x2) // 2, (y1 + y2) // 2)
        history = [(round(hx * sx), round(hy * sy)) for (hx, hy) in getattr(p, "history", [])]
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
    current_user: User = Depends(require_admin),
) -> FeedSessionResponse:
    if req.device_id:
        with _feeds_lock:
            for fid, did in list(_feed_device_ids.items()):
                if did == req.device_id and fid in _active_feeds:
                    logger.info("Device %s already streaming under feed %s", req.device_id, fid)
                    return _build_feed_response(_active_feeds[fid], req.source_type)
            if req.device_id in _starting_devices:
                pending_fid = _starting_devices[req.device_id]
                logger.info("Device %s is already starting under feed %s", req.device_id, pending_fid)
                if pending_fid in _active_feeds:
                    return _build_feed_response(_active_feeds[pending_fid], req.source_type)
                return FeedSessionResponse(
                    feed_id=pending_fid,
                    source_type=req.source_type,
                    status="starting",
                    description=f"Device {req.device_name or req.device_id} starting",
                    is_live=True,
                    frames_produced=0,
                    error_count=0,
                    started_at=time.time(),
                    is_recording=True,
                    device_id=req.device_id,
                    device_name=req.device_name,
                )
            feed_id = f"feed_{uuid.uuid4().hex[:8]}"
            _starting_devices[req.device_id] = feed_id
    else:
        feed_id = f"feed_{uuid.uuid4().hex[:8]}"

    try:
        source = await asyncio.to_thread(_resolve_source, req.source_type, req.source_uri)
    except Exception:
        if req.device_id:
            with _feeds_lock:
                _starting_devices.pop(req.device_id, None)
        raise

    settings = get_settings()
    with _feeds_lock:
        _feed_peaks[feed_id] = 0.0

    # `on_fast_result`/`on_density_result` run on RealtimeFeedSession's own
    # background threads, never on the asyncio event loop -- capture the
    # real loop here (this coroutine IS running on it) so both callbacks can
    # hand a broadcast back to it with `run_coroutine_threadsafe`. Calling
    # `asyncio.get_running_loop()` from those background threads directly
    # would always raise (no running loop on that thread).
    loop = asyncio.get_running_loop()
    channel = connection_manager.feed_channel(feed_id)
    sys_channel = connection_manager.system_channel()

    def broadcast(envelope: WSEnvelope) -> None:
        asyncio.run_coroutine_threadsafe(connection_manager.broadcast(channel, envelope), loop)

    def broadcast_system(envelope: WSEnvelope) -> None:
        asyncio.run_coroutine_threadsafe(connection_manager.broadcast(sys_channel, envelope), loop)

    def on_source_status_change(status: str, reason: str) -> None:
        # Runs on the producer thread the moment the failover source switches.
        data = {"source_status": status, "source_status_reason": reason, "source_status_changed_at": time.time()}
        broadcast(WSEnvelope(type=WSEventType.FEED_PROGRESS, job_id=feed_id, data=data))
        broadcast_system(WSEnvelope(type=WSEventType.SYSTEM_METRICS, job_id=feed_id, data={"feed_id": feed_id, **data}))

    fallback_uri = req.fallback_uri if req.fallback_uri is not None else drone_fallback_uri(req.device_id, feed_id)
    source = _wrap_with_failover(source, fallback_uri, on_source_status_change)

    # UI frame transmission is decoupled from the fast loop: detection,
    # tracking and analytics run on every frame, but annotated JPEGs are only
    # rendered/encoded/sent at most `feed_ws_max_fps` times a second.
    min_send_interval_s = 1.0 / settings.feed_ws_max_fps if settings.feed_ws_max_fps > 0 else 0.0
    last_frame_sent_at = 0.0

    def on_fast_result(fast: FastResult) -> None:
        nonlocal last_frame_sent_at
        now = time.monotonic()
        if now - last_frame_sent_at < min_send_interval_s:
            return
        last_frame_sent_at = now
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

            with _feeds_lock:
                current_peak = _feed_peaks.get(feed_id, 0.0)
                recorder = _feed_recorders.get(feed_id)

            annotated = render_video_overlay(
                disp_frame,
                disp_detections,
                frame_number=fast.frame.frame_number,
                total_frames=None,
                video_time_s=fast.frame.timestamp_s,
                total_duration_s=None,
                visible_count=fast.visible_count,
                estimated_count=_latest_density_count(feed_id),
                peak_count=current_peak,
                fps=round(1000.0 / fast.latency_ms, 2) if fast.latency_ms > 0 else None,
                latency_ms=fast.latency_ms,
                density=_latest_density_value(feed_id),
                cluster_count=_latest_density_clusters(feed_id),
                largest_cluster=_latest_density_largest_cluster(feed_id),
                risk_status=_latest_density_risk_status(feed_id),
                risk_score=_latest_density_risk_score(feed_id),
            )
            if recorder is not None:
                recorder.write_detection_frame(annotated)
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
                # Repeated on every frame so a client that joins mid-outage
                # still learns the source state without waiting for a switch.
                "source_status": getattr(source, "source_status", "LIVE"),
            }
            with _feeds_lock:
                _feed_latest_frames.setdefault(feed_id, {})["frame_jpeg"] = frame_jpeg
            broadcast(WSEnvelope(type=WSEventType.FEED_PROGRESS, job_id=feed_id, data=data))
        except Exception:
            logger.exception("Error building/broadcasting fast-path frame for feed %s", feed_id)

    def on_density_result(density: DensityResult) -> None:
        try:
            r = density.result
            count = round(float(r.inference.count), 2)
            with _feeds_lock:
                peak = max(_feed_peaks.get(feed_id, 0.0), count)
                _feed_peaks[feed_id] = peak
                active_session = _active_feeds.get(feed_id)
                recorder = _feed_recorders.get(feed_id)
            dcomp = float(r.risk.components.get("relative_density_per_100px2", 0.0) or 0.0)

            frame_bgr = density.frame.frame_bgr
            fh, fw = frame_bgr.shape[:2]
            max_side = settings.feed_frame_max_side
            density_map = r.inference.density_map
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

            if recorder is not None:
                recorder.save_snapshot(heatmap_frame, "heatmap")
                recorder.save_snapshot(combined_frame, "combined")

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
                "cluster_count": r.clusters.n_clusters,
                "largest_cluster": r.clusters.largest_cluster_size,
                "risk_status": r.risk.status,
                "risk_score": round(float(r.risk.score), 2),
                "density_latency_ms": round(density.latency_ms, 2),
                "heatmap_jpeg": heatmap_jpeg,
                "combined_jpeg": combined_jpeg,
            }
            with _feeds_lock:
                _feed_latest_frames.setdefault(feed_id, {})["heatmap_jpeg"] = heatmap_jpeg
                _feed_latest_frames.setdefault(feed_id, {})["combined_jpeg"] = combined_jpeg
                saved_frame_jpeg = _feed_latest_frames.get(feed_id, {}).get("frame_jpeg")
            broadcast(WSEnvelope(type=WSEventType.FEED_PROGRESS, job_id=feed_id, data=data))

            # Broadcast to system channel for Overview dashboard real-time synchronization
            fast_res = active_session.latest_fast_result if active_session else None
            system_data = {
                "feed_id": feed_id,
                "frame_number": fast_res.frame.frame_number if fast_res else 0,
                "timestamp_s": round(fast_res.frame.timestamp_s, 2) if fast_res else 0.0,
                "estimated_count": count,
                "visible_count": fast_res.visible_count if fast_res else int(count),
                "peak_count": round(peak, 2),
                "density": round(dcomp, 3),
                "cluster_count": r.clusters.n_clusters,
                "largest_cluster": r.clusters.largest_cluster_size,
                "risk_status": r.risk.status,
                "risk_score": round(float(r.risk.score), 2),
                "latency_ms": round(fast_res.latency_ms, 2) if fast_res else 0.0,
                "fps": round(1000.0 / fast_res.latency_ms, 2) if fast_res and fast_res.latency_ms > 0 else 30.0,
                "frame_jpeg": saved_frame_jpeg,
            }
            broadcast_system(WSEnvelope(type=WSEventType.SYSTEM_METRICS, job_id=feed_id, data=system_data))
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
        if err.stage == "producer":
            asyncio.run_coroutine_threadsafe(
                _finalize_feed_session(feed_id, connection_manager, reason="source_error"), loop
            )

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
        detection_stride=settings.feed_detection_stride,
    )

    try:
        # Opening a network stream can block for the FFmpeg open timeout (and
        # a failover start may then open the backup) -- keep the loop free.
        await asyncio.to_thread(session.start)
    except Exception as exc:
        with _feeds_lock:
            _feed_peaks.pop(feed_id, None)
            if req.device_id:
                _starting_devices.pop(req.device_id, None)
        raise HTTPException(status_code=500, detail=f"Failed to start feed: {exc}") from exc

    recorder = LiveFeedRecorder(feed_id)
    with _feeds_lock:
        _active_feeds[feed_id] = session
        _feed_recorders[feed_id] = recorder
        if req.device_id:
            _feed_device_ids[feed_id] = req.device_id
            _starting_devices.pop(req.device_id, None)
        if req.device_name:
            _feed_device_names[feed_id] = req.device_name
    logger.info("Auto-recording initialized for feed %s (session %s)", feed_id, recorder.session_uid)
    return _build_feed_response(session, req.source_type)


def _latest_density(feed_id: str) -> Optional[DensityResult]:
    with _feeds_lock:
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
    return d.result.clusters.n_clusters if d else 0


def _latest_density_largest_cluster(feed_id: str) -> int:
    d = _latest_density(feed_id)
    return d.result.clusters.largest_cluster_size if d else 0


def _latest_density_risk_status(feed_id: str) -> str:
    d = _latest_density(feed_id)
    return d.result.risk.status if d else "SAFE"


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
    current_user: User = Depends(require_admin),
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
    with _feeds_lock:
        session = _active_feeds.get(res.feed_id)
    if session is not None:
        setattr(session, "_temp_upload_path", tmp_path)
    return res


@router.get("", response_model=List[FeedSessionResponse])
async def list_feeds(
    current_user: User = Depends(get_current_user),
) -> List[FeedSessionResponse]:
    with _feeds_lock:
        sessions = list(_active_feeds.values())
    return [_build_feed_response(sess, "feed") for sess in sessions]


@router.get("/{feed_id}", response_model=FeedSessionResponse)
async def get_feed(
    feed_id: str,
    current_user: User = Depends(get_current_user),
) -> FeedSessionResponse:
    with _feeds_lock:
        sess = _active_feeds.get(feed_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"Feed '{feed_id}' not found")
    return _build_feed_response(sess, "feed")


@router.post("/{feed_id}/stop", response_model=FeedSessionResponse)
async def stop_feed(
    feed_id: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager),
    current_user: User = Depends(require_admin),
) -> FeedSessionResponse:
    with _feeds_lock:
        sess = _active_feeds.get(feed_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"Feed '{feed_id}' not found")

    await _finalize_feed_session(feed_id, connection_manager, reason="user_stop")
    return _build_feed_response(sess, "feed")


@router.post("/{feed_id}/source", response_model=FeedSessionResponse)
async def set_feed_source_mode(
    feed_id: str,
    req: FeedSourceModeRequest,
    current_user: User = Depends(require_admin),
) -> FeedSessionResponse:
    """Manual override for drone failover: 'backup' forces the backup
    archive, 'auto' returns to automatic failover/recovery."""
    with _feeds_lock:
        sess = _active_feeds.get(feed_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"Feed '{feed_id}' not found")
    if not isinstance(sess.source, FailoverFrameSource):
        raise HTTPException(status_code=409, detail="This feed has no backup source configured")
    sess.source.set_mode(MODE_BACKUP if req.mode == "backup" else MODE_AUTO)
    return _build_feed_response(sess, "feed")


async def _finalize_feed_session(
    feed_id: str, connection_manager: ConnectionManager, reason: str = "stopped"
) -> None:
    with _feeds_lock:
        sess = _active_feeds.pop(feed_id, None)
        _feed_peaks.pop(feed_id, None)
        _feed_latest_frames.pop(feed_id, None)
        dev_id = _feed_device_ids.pop(feed_id, None)
        if dev_id:
            _starting_devices.pop(dev_id, None)
        for did, fid in list(_starting_devices.items()):
            if fid == feed_id:
                _starting_devices.pop(did, None)
        _feed_device_names.pop(feed_id, None)
        recorder = _feed_recorders.pop(feed_id, None)
    if sess is None:
        return
    try:
        await asyncio.to_thread(sess.stop)
    except Exception:
        logger.exception("Error stopping session for feed %s", feed_id)
    if recorder is not None:
        await asyncio.to_thread(recorder.finalize)
    tmp_path = getattr(sess, "_temp_upload_path", None)
    if tmp_path is not None and isinstance(tmp_path, Path):
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    response = _build_feed_response(sess, "feed")
    await connection_manager.broadcast(
        connection_manager.feed_channel(feed_id),
        WSEnvelope(
            type=WSEventType.FEED_STATUS,
            job_id=feed_id,
            data={"feed_id": feed_id, "status": "stopped", "reason": reason},
        ),
    )
    await connection_manager.broadcast(
        connection_manager.system_channel(),
        WSEnvelope(
            type=WSEventType.SYSTEM_METRICS,
            job_id=feed_id,
            data={"feed_id": feed_id, "status": "stopped"},
        ),
    )


@router.post("/{feed_id}/record/start")
async def start_feed_recording(
    feed_id: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager),
    current_user: User = Depends(require_admin),
):
    """Start recording detection-view video and snapshots for an active feed session."""
    with _feeds_lock:
        session = _active_feeds.get(feed_id)
        if session is None or not session.is_running:
            raise HTTPException(status_code=404, detail=f"Live feed '{feed_id}' is not active or running")

        if feed_id in _feed_recorders:
            rec = _feed_recorders[feed_id]
            dur = round(time.time() - rec._start_wall, 1) if rec._start_wall is not None else 0.0
            return {
                "status": "already_recording",
                "feed_id": feed_id,
                "recording": True,
                "session_uid": rec.session_uid,
                "started_at": rec._start_wall,
                "duration_s": dur,
            }

        recorder = LiveFeedRecorder(feed_id)
        _feed_recorders[feed_id] = recorder
    logger.info("Manual recording started for feed %s (session %s)", feed_id, recorder.session_uid)

    await connection_manager.broadcast(
        connection_manager.feed_channel(feed_id),
        WSEnvelope(
            type=WSEventType.FEED_STATUS,
            job_id=feed_id,
            data={
                "feed_id": feed_id,
                "event": "recording_started",
                "recording": True,
                "session_uid": recorder.session_uid,
            },
        ),
    )

    return {
        "status": "started",
        "feed_id": feed_id,
        "recording": True,
        "session_uid": recorder.session_uid,
        "started_at": recorder._start_wall,
        "message": "Live feed recording started",
    }


@router.post("/{feed_id}/record/stop")
async def stop_feed_recording(
    feed_id: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager),
    current_user: User = Depends(require_admin),
):
    """Stop active recording for a feed session, finalize output, and upload to S3 if configured."""
    with _feeds_lock:
        recorder = _feed_recorders.pop(feed_id, None)
    if recorder is None:
        raise HTTPException(status_code=400, detail=f"No active recording for feed '{feed_id}'")

    summary = await asyncio.to_thread(recorder.finalize)
    logger.info("Manual recording stopped for feed %s (summary: %s)", feed_id, summary)

    await connection_manager.broadcast(
        connection_manager.feed_channel(feed_id),
        WSEnvelope(
            type=WSEventType.FEED_STATUS,
            job_id=feed_id,
            data={
                "feed_id": feed_id,
                "event": "recording_stopped",
                "recording": False,
                "summary": summary,
            },
        ),
    )

    return {
        "status": "stopped",
        "feed_id": feed_id,
        "recording": False,
        "summary": summary,
        "message": "Live feed recording stopped and saved",
    }


@router.get("/{feed_id}/record/status")
async def get_feed_recording_status(
    feed_id: str,
    current_user: User = Depends(get_current_user),
):
    """Check whether a feed is currently being recorded."""
    with _feeds_lock:
        session = _active_feeds.get(feed_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Feed '{feed_id}' not found")
        recorder = _feed_recorders.get(feed_id)
    if recorder is None:
        return {
            "feed_id": feed_id,
            "recording": False,
            "session_uid": None,
            "duration_s": 0,
            "frames": 0,
        }
    is_actively_recording = recorder._start_wall is not None
    duration = round(time.time() - recorder._start_wall, 1) if recorder._start_wall is not None else 0.0
    return {
        "feed_id": feed_id,
        "recording": is_actively_recording,
        "session_uid": recorder.session_uid,
        "duration_s": duration,
        "frames": recorder._frame_count,
    }


@router.get("/recordings/{session_uid}/video")
async def get_recording_video(
    session_uid: str,
    current_user: User = Depends(get_current_user),
):
    """Serve or redirect to a finished live recording MP4."""
    import re
    if not re.match(r"^[a-zA-Z0-9_\-]+$", session_uid):
        raise HTTPException(status_code=400, detail="Invalid session UID format")

    local_path = LIVE_RECORDINGS_DIR / f"{session_uid}_detection.mp4"
    if not is_within(local_path, LIVE_RECORDINGS_DIR):
        raise HTTPException(status_code=403, detail="Access denied")

    # If local file exists, serve it directly
    if local_path.exists():
        return FileResponse(path=str(local_path), media_type="video/mp4", filename=f"{session_uid}.mp4")

    # Otherwise if stored in S3, redirect to presigned URL
    key = f"live_feeds/{session_uid}/detection.mp4"
    if storage.is_s3_enabled():
        try:
            return RedirectResponse(
                url=storage.presigned_url(key),
                status_code=307,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        except Exception:
            logger.exception("Failed to generate presigned S3 URL for live recording %s", session_uid)

    raise HTTPException(status_code=404, detail=f"Recording '{session_uid}' not found")
