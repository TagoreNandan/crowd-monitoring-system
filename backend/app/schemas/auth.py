"""Pydantic schemas for User Management and Admin Activity Logging."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_serializer



class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = Field("", max_length=100)
    role: Optional[str] = Field("analyst", pattern="^(admin|analyst)$")


class UserLoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str
    full_name: Optional[str] = ""


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    full_name: Optional[str] = ""
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

    @field_serializer("created_at", "last_login", check_fields=False)
    def serialize_dt(self, dt: Optional[datetime], _info):
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()


class UserUpdateRequest(BaseModel):
    is_active: Optional[bool] = None
    role: Optional[str] = None
    full_name: Optional[str] = None


class ActivityLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: Optional[int] = None
    username: str
    action: str
    category: str
    details: Optional[str] = ""
    severity: str
    timestamp: datetime

    @field_serializer("timestamp", check_fields=False)
    def serialize_timestamp(self, dt: datetime, _info):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()


class AdminOverviewStats(BaseModel):
    total_users: int
    active_users: int
    admin_users: int
    total_activities: int
    database_engine: str = "sqlite"
    database_status: str = "ONLINE"
    recent_activities: List[ActivityLogResponse]
