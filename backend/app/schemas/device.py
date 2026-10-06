"""Pydantic schemas for Device Management & Stream Onboarding."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_serializer


class DeviceCreateRequest(BaseModel):
    device_id: str = Field(..., min_length=2, max_length=50, description="Unique camera/device identifier (e.g., CAM-GATE-01)")
    name: str = Field(..., min_length=2, max_length=100, description="Human readable device name")
    description: Optional[str] = Field("", max_length=500)
    location: str = Field(..., min_length=2, max_length=100, description="Physical installation location (e.g., North Concourse)")
    stream_type: str = Field("RTSP", description="Stream protocol: RTSP, WEBCAM, HTTP_HLS, VIDEO_FILE")
    stream_url: str = Field(..., min_length=1, max_length=500, description="Stream URL, RTSP endpoint, or device index")
    status: Optional[str] = Field("ONLINE", description="Device operational status: ONLINE, STANDBY, OFFLINE, MAINTENANCE")
    fps: Optional[float] = Field(None, description="Reported or configured frames per second")
    resolution: Optional[str] = Field("", description="Detected video resolution (e.g., 1920x1080)")


class DeviceUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    description: Optional[str] = None
    location: Optional[str] = Field(None, min_length=2, max_length=100)
    stream_type: Optional[str] = None
    stream_url: Optional[str] = Field(None, min_length=1, max_length=500)
    status: Optional[str] = None
    fps: Optional[float] = None
    resolution: Optional[str] = None


class DeviceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    device_id: str
    name: str
    description: Optional[str] = ""
    location: str
    stream_type: str
    stream_url: str
    status: str
    fps: Optional[float] = None
    resolution: Optional[str] = ""
    created_by: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    @field_serializer("created_at", "updated_at", check_fields=False)
    def serialize_dt(self, dt: Optional[datetime], _info):
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()


class StreamProbeRequest(BaseModel):
    stream_url: str = Field(..., min_length=1, description="URL or device index to probe")
    stream_type: str = Field("RTSP", description="Protocol type: RTSP, WEBCAM, HTTP_HLS, VIDEO_FILE")


class StreamProbeResponse(BaseModel):
    is_reachable: bool
    resolution: Optional[str] = None
    fps: Optional[float] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
