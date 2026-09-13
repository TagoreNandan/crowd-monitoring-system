"""SQLAlchemy ORM models for Users, System Activities, and Audit logs."""

from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
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
    category = Column(String(50), nullable=False, default="GENERAL")  # "AUTH", "INFERENCE", "SYSTEM"
    details = Column(Text, nullable=True, default="")
    severity = Column(String(20), nullable=False, default="INFO")  # "INFO", "WARNING", "CRITICAL"
    timestamp = Column(DateTime, default=utcnow, nullable=False, index=True)
