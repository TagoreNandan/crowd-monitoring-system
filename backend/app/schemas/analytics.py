"""Pydantic schemas for Multi-Stream Live Analytics Dashboard."""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class StreamAnalyticsSnapshot(BaseModel):
    feed_id: str
    device_id: Optional[str] = None
    camera_name: str
    location: str
    status: str
    is_live: bool
    estimated_count: float
    visible_count: Optional[int] = None
    peak_count: float
    risk_status: str
    risk_score: float
    fps: Optional[float] = None
    latency_ms: Optional[float] = None
    frame_jpeg: Optional[str] = None
    heatmap_jpeg: Optional[str] = None
    combined_jpeg: Optional[str] = None


class ZoneDistributionMetric(BaseModel):
    zone_name: str
    camera_count: int
    crowd_count: float
    risk_status: str
    density_percent: float = Field(0.0, description="Relative density compared to max zone capacity")


class TimelineDataPoint(BaseModel):
    timestamp: str
    total_crowd: float
    visible_count: int
    active_streams: int


class AnalyticsAlert(BaseModel):
    id: str
    timestamp: str
    zone: str
    camera_name: str
    severity: str  # "INFO", "WARNING", "CRITICAL"
    message: str
    crowd_count: float


class LiveAnalyticsResponse(BaseModel):
    total_crowd: float
    total_visible: int
    peak_crowd: float
    fleet_risk_status: str
    fleet_risk_score: float
    active_cameras_count: int
    total_cameras_count: int
    streams: List[StreamAnalyticsSnapshot]
    zones: List[ZoneDistributionMetric]
    timeline: List[TimelineDataPoint]
    recent_alerts: List[AnalyticsAlert]
    updated_at: str
