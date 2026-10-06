"""FastAPI authentication dependencies and audit logging helpers."""

from __future__ import annotations

import logging
import json
from typing import Optional

from fastapi import Depends, HTTPException, Query, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from ..db.models import ActivityLog, User
from ..db.session import get_db
from .security import decode_access_token

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token", auto_error=False)


def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    token_query: Optional[str] = Query(None, alias="token"),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Returns the authenticated user or None if no token or invalid token provided."""
    resolved_token = token or token_query
    if not resolved_token:
        return None
    payload = decode_access_token(resolved_token)
    if not payload:
        return None
    username = payload.get("sub")
    if not username:
        return None
    user = db.query(User).filter(User.username == username).first()
    return user


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    token_query: Optional[str] = Query(None, alias="token"),
    db: Session = Depends(get_db),
) -> User:
    """Dependency that requires a valid JWT token from Authorization header or ?token= query parameter."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    resolved_token = token or token_query
    if not resolved_token:
        raise credentials_exception
    payload = decode_access_token(resolved_token)
    if not payload:
        raise credentials_exception
    username: str | None = payload.get("sub")
    if not username:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account",
        )
    return user


def require_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    """Dependency that restricts endpoint access strictly to admin users."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for this action",
        )
    return current_user


def log_activity(
    db: Session,
    action: str,
    category: str = "GENERAL",
    details: str = "",
    severity: str = "INFO",
    user: Optional[User] = None,
    username: Optional[str] = None,
    job_id: Optional[str] = None,
    job_type: Optional[str] = None,
    output_keys: Optional[dict[str, str]] = None,
    storage_backend: Optional[str] = None,
) -> ActivityLog:
    """Record an action in the ActivityLog table."""
    try:
        user_id = user.id if user else None
        uname = username or (user.username if user else "system")
        entry = ActivityLog(
            user_id=user_id,
            username=uname,
            action=action,
            category=category,
            details=details,
            severity=severity,
            job_id=job_id,
            job_type=job_type,
            output_keys=json.dumps(output_keys) if output_keys is not None else None,
            storage_backend=storage_backend,
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        return entry
    except Exception as e:
        logger.error(f"Failed to record activity log: {e}")
        db.rollback()
        return None
