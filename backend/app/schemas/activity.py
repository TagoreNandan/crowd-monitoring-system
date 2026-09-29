"""Schemas for role-scoped activity history and output downloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, field_serializer


class ActivityItem(BaseModel):
    id: int
    username: str
    action: str
    category: str
    details: Optional[str] = ""
    severity: str
    timestamp: datetime
    job_id: Optional[str] = None
    job_type: Optional[str] = None
    has_download: bool
    available_views: List[str] = Field(default_factory=list)

    @field_serializer("timestamp")
    def serialize_timestamp(self, value: datetime, _info):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()


class ActivityListResponse(BaseModel):
    items: List[ActivityItem]
    total: int


class ActivityDownloadResponse(BaseModel):
    mode: str
    url: str
    expires_in: Optional[int] = None
    expires_at: Optional[int] = None
    filename: str
    view: str
