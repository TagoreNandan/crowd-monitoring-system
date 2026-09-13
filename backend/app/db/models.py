"""SQLAlchemy ORM models for Users, System Activities, and Audit logs."""

from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from .session import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=True, default="")
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="user")  # "admin" | "user"
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)


class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, nullable=True, index=True)
    username = Column(String(50), nullable=False, default="system")
    action = Column(String(100), nullable=False, index=True)  # e.g., "USER_LOGIN", "IMAGE_ANALYSIS"
    category = Column(String(50), nullable=False, default="GENERAL")  # "AUTH", "INFERENCE", "SYSTEM", "DEVICES"
    details = Column(Text, nullable=True, default="")
    severity = Column(String(20), nullable=False, default="INFO")  # "INFO", "WARNING", "CRITICAL"
    timestamp = Column(DateTime, default=utcnow, nullable=False, index=True)


class Device(Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    device_id = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True, default="")
    location = Column(String(100), nullable=False)
    stream_type = Column(String(30), nullable=False, default="RTSP")  # RTSP, WEBCAM, HTTP_HLS, VIDEO_FILE
    stream_url = Column(String(500), nullable=False)
    status = Column(String(20), nullable=False, default="ONLINE")  # ONLINE, STANDBY, OFFLINE, MAINTENANCE
    fps = Column(Float, nullable=True)
    resolution = Column(String(50), nullable=True, default="")
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)
