"""Device Management & Stream Onboarding Router.

Allows administrators to enroll, monitor, test, and decommission camera streams.
Authenticated operators can view the catalog of active devices and launch real-time
crowd monitoring feeds.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, List, Optional

import cv2
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from ..auth.dependencies import get_current_user, log_activity, require_admin
from ..db.models import Device, User
from ..db.session import get_db
from ..schemas.device import (
    DeviceCreateRequest,
    DeviceResponse,
    DeviceUpdateRequest,
    StreamProbeRequest,
    StreamProbeResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/devices", tags=["devices"])


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


def _probe_stream_sync(stream_url: str, stream_type: str) -> dict:
    """Synchronous worker to test OpenCV stream readability."""
    t0 = time.time()
    cap = None
    try:
        if "youtube.com" in stream_url or "youtu.be" in stream_url:
            source: Any = _resolve_youtube_stream_url(stream_url)
        elif stream_url.isdigit():
            source = int(stream_url)
        else:
            source = stream_url

        cap = cv2.VideoCapture(source)

        # Set connection timeouts if supported by OpenCV backend
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
        except Exception:
            pass

        if not cap.isOpened():
            latency = (time.time() - t0) * 1000
            return {
                "is_reachable": False,
                "error": "Could not connect to camera/stream source",
                "latency_ms": round(latency, 1),
            }

        ret, frame = cap.read()
        latency = (time.time() - t0) * 1000

        if not ret or frame is None:
            return {
                "is_reachable": False,
                "error": "Stream opened but failed to retrieve video frame",
                "latency_ms": round(latency, 1),
            }

        h, w = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1 or fps > 240:
            fps = 30.0

        return {
            "is_reachable": True,
            "resolution": f"{w}x{h}",
            "fps": round(float(fps), 1),
            "latency_ms": round(latency, 1),
            "error": None,
        }
    except Exception as exc:
        latency = (time.time() - t0) * 1000
        return {
            "is_reachable": False,
            "error": str(exc),
            "latency_ms": round(latency, 1),
        }
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


@router.get("", response_model=List[DeviceResponse])
def list_devices(
    search: Optional[str] = Query(None, description="Search by name, ID, or location"),
    status_filter: Optional[str] = Query(None, alias="status"),
    stream_type: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List registered camera devices. Accessible to all authenticated operators."""
    query = db.query(Device)

    if status_filter and status_filter.upper() != "ALL":
        query = query.filter(func.upper(Device.status) == status_filter.upper())

    if stream_type and stream_type.upper() != "ALL":
        query = query.filter(func.upper(Device.stream_type) == stream_type.upper())

    if search:
        pattern = f"%{search.strip().lower()}%"
        query = query.filter(
            func.lower(Device.name).like(pattern)
            | func.lower(Device.device_id).like(pattern)
            | func.lower(Device.location).like(pattern)
        )

    return query.order_by(desc(Device.created_at)).all()


@router.get("/{device_id}", response_model=DeviceResponse)
def get_device(
    device_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve details of a single device by ID."""
    device = db.query(Device).filter(Device.device_id == device_id).first()
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device '{device_id}' not found",
        )
    return device


@router.post("/probe", response_model=StreamProbeResponse)
async def probe_stream(
    req: StreamProbeRequest,
    admin: User = Depends(require_admin),
):
    """Admin-only: Probes a candidate stream URL to verify connectivity and report resolution/fps."""
    try:
        res = await asyncio.wait_for(
            asyncio.to_thread(_probe_stream_sync, req.stream_url, req.stream_type),
            timeout=5.0,
        )
        return StreamProbeResponse(**res)
    except asyncio.TimeoutError:
        return StreamProbeResponse(
            is_reachable=False,
            latency_ms=5000.0,
            error="Connection probe timed out (stream host unreachable)",
        )
    except Exception as exc:
        return StreamProbeResponse(
            is_reachable=False,
            error=str(exc),
        )


@router.post("", response_model=DeviceResponse, status_code=status.HTTP_201_CREATED)
async def create_device(
    req: DeviceCreateRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin-only: Enroll a new camera device into the fleet."""
    clean_id = req.device_id.strip().upper()
    existing = db.query(Device).filter(Device.device_id == clean_id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Device identifier '{clean_id}' is already registered",
        )

    resolution = req.resolution or ""
    fps = req.fps
    device_status = req.status or "ONLINE"

    # Attempt a quick non-blocking probe to verify stream if resolution not provided
    if not resolution:
        try:
            probe = await asyncio.wait_for(
                asyncio.to_thread(_probe_stream_sync, req.stream_url, req.stream_type),
                timeout=3.0,
            )
            if probe.get("is_reachable"):
                resolution = probe.get("resolution") or ""
                fps = probe.get("fps") or fps
                device_status = "ONLINE"
            else:
                # Still allow enrollment but mark as STANDBY if unreachable
                device_status = "STANDBY"
        except Exception:
            device_status = "STANDBY"

    device = Device(
        device_id=clean_id,
        name=req.name.strip(),
        description=req.description or "",
        location=req.location.strip(),
        stream_type=req.stream_type.upper(),
        stream_url=req.stream_url.strip(),
        status=device_status,
        fps=fps,
        resolution=resolution,
        created_by=admin.id,
    )
    db.add(device)
    db.commit()
    db.refresh(device)

    log_activity(
        db,
        action="DEVICE_ONBOARDED",
        category="DEVICES",
        details=f"Camera {device.name} ({device.device_id}) enrolled at {device.location}. Stream: {device.stream_type}",
        severity="INFO",
        user=admin,
    )

    return device


@router.put("/{device_id}", response_model=DeviceResponse)
def update_device(
    device_id: str,
    req: DeviceUpdateRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin-only: Update device metadata, status, or stream URL."""
    device = db.query(Device).filter(Device.device_id == device_id).first()
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device '{device_id}' not found",
        )

    if req.name is not None:
        device.name = req.name.strip()
    if req.description is not None:
        device.description = req.description
    if req.location is not None:
        device.location = req.location.strip()
    if req.stream_type is not None:
        device.stream_type = req.stream_type.upper()
    if req.stream_url is not None:
        device.stream_url = req.stream_url.strip()
    if req.status is not None:
        device.status = req.status.upper()
    if req.fps is not None:
        device.fps = req.fps
    if req.resolution is not None:
        device.resolution = req.resolution

    db.commit()
    db.refresh(device)

    log_activity(
        db,
        action="DEVICE_UPDATED",
        category="DEVICES",
        details=f"Device {device.device_id} ({device.name}) updated. Status: {device.status}",
        severity="INFO",
        user=admin,
    )

    return device


@router.delete("/{device_id}", status_code=status.HTTP_200_OK)
def delete_device(
    device_id: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin-only: Decommission and remove a camera device from the fleet."""
    device = db.query(Device).filter(Device.device_id == device_id).first()
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device '{device_id}' not found",
        )

    dev_name = device.name
    db.delete(device)
    db.commit()

    log_activity(
        db,
        action="DEVICE_DECOMMISSIONED",
        category="DEVICES",
        details=f"Device {device_id} ({dev_name}) removed from system fleet.",
        severity="WARNING",
        user=admin,
    )

    return {"message": f"Device {device_id} successfully decommissioned", "device_id": device_id}


@router.post("/{device_id}/test", response_model=StreamProbeResponse)
async def test_device(
    device_id: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin-only: Test live connectivity for an enrolled device and update status."""
    device = db.query(Device).filter(Device.device_id == device_id).first()
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device '{device_id}' not found",
        )

    try:
        res = await asyncio.wait_for(
            asyncio.to_thread(_probe_stream_sync, device.stream_url, device.stream_type),
            timeout=5.0,
        )
        probe = StreamProbeResponse(**res)
        if probe.is_reachable:
            device.status = "ONLINE"
            if probe.resolution:
                device.resolution = probe.resolution
            if probe.fps:
                device.fps = probe.fps
        else:
            device.status = "OFFLINE"

        db.commit()
        db.refresh(device)
        return probe
    except asyncio.TimeoutError:
        device.status = "OFFLINE"
        db.commit()
        return StreamProbeResponse(
            is_reachable=False,
            latency_ms=5000.0,
            error="Connection probe timed out (stream host unreachable)",
        )
    except Exception as exc:
        device.status = "ERROR"
        db.commit()
        return StreamProbeResponse(is_reachable=False, error=str(exc))
