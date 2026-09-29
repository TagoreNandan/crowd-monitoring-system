"""Role-scoped activity history and analysis-output download metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, or_
from sqlalchemy.orm import Session

from .. import storage
from ..auth.dependencies import get_current_user, log_activity
from ..db.models import ActivityLog, User
from ..db.session import get_db
from ..dependencies import get_job_store
from ..jobs.store import JobStore
from ..schemas.activity import ActivityDownloadResponse, ActivityItem, ActivityListResponse

router = APIRouter(prefix="/api/activity", tags=["activity"])


def _parse_output_keys(entry: ActivityLog) -> dict[str, str]:
    if not entry.output_keys:
        return {}
    try:
        parsed: Any = json.loads(entry.output_keys)
    except (TypeError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(view): str(key) for view, key in parsed.items() if key}


def _to_activity_item(entry: ActivityLog) -> ActivityItem:
    output_keys = _parse_output_keys(entry)
    return ActivityItem(
        id=entry.id,
        username=entry.username,
        action=entry.action,
        category=entry.category,
        details=entry.details,
        severity=entry.severity,
        timestamp=entry.timestamp,
        job_id=entry.job_id,
        job_type=entry.job_type,
        has_download=bool(output_keys),
        available_views=list(output_keys),
    )


def _get_visible_entry(activity_id: int, current_user: User, db: Session) -> ActivityLog:
    query = db.query(ActivityLog).filter(ActivityLog.id == activity_id)
    if current_user.role != "admin":
        query = query.filter(ActivityLog.user_id == current_user.id)
    entry = query.first()
    if entry is None:
        raise HTTPException(status_code=404, detail="Activity log not found")
    return entry


@router.get("", response_model=ActivityListResponse)
def list_activity(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    category: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    has_download: Optional[bool] = Query(None),
    username: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ActivityListResponse:
    query = db.query(ActivityLog)
    if current_user.role != "admin":
        query = query.filter(ActivityLog.user_id == current_user.id)
    elif username:
        query = query.filter(ActivityLog.username == username)
    if category:
        query = query.filter(ActivityLog.category == category)
    if action:
        query = query.filter(ActivityLog.action == action)
    if severity:
        query = query.filter(ActivityLog.severity == severity)
    if has_download is True:
        query = query.filter(
            ActivityLog.output_keys.is_not(None),
            ActivityLog.output_keys != "",
            ActivityLog.output_keys != "{}",
        )
    elif has_download is False:
        query = query.filter(
            or_(
                ActivityLog.output_keys.is_(None),
                ActivityLog.output_keys == "",
                ActivityLog.output_keys == "{}",
            )
        )

    total = query.count()
    entries = (
        query.order_by(desc(ActivityLog.timestamp), desc(ActivityLog.id))
        .offset(offset)
        .limit(limit)
        .all()
    )
    return ActivityListResponse(items=[_to_activity_item(entry) for entry in entries], total=total)


@router.get("/{activity_id}/download", response_model=ActivityDownloadResponse)
def download_activity_output(
    activity_id: int,
    view: str = Query(..., min_length=1),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    job_store: JobStore = Depends(get_job_store),
) -> ActivityDownloadResponse:
    entry = _get_visible_entry(activity_id, current_user, db)
    output_keys = _parse_output_keys(entry)
    key = output_keys.get(view)
    if key is None:
        raise HTTPException(status_code=404, detail="Requested output view is not available")

    if entry.storage_backend == "s3":
        filename = Path(key).name or f"{entry.job_id}_{view}"
        try:
            if not storage.object_exists(key):
                raise HTTPException(status_code=410, detail="The requested output is no longer available")
            url_info = storage.get_presigned_url_info(key, response_disposition=filename)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail="The output storage service is temporarily unavailable") from exc
        response = ActivityDownloadResponse(
            mode="s3",
            url=url_info["url"],
            expires_in=url_info["expires_in"],
            expires_at=url_info["expires_at"],
            filename=filename,
            view=view,
        )
    else:
        if not entry.job_id or job_store.get(entry.job_id) is None:
            raise HTTPException(status_code=410, detail="The requested output is no longer available")
        if entry.job_type == "VIDEO":
            url = f"/api/jobs/{entry.job_id}/video/{view}"
            extension = "mp4"
        else:
            url = f"/api/jobs/jobs/{entry.job_id}/image/{view}"
            extension = "jpg"
        response = ActivityDownloadResponse(
            mode="local",
            url=url,
            expires_in=None,
            expires_at=None,
            filename=f"{entry.job_id}_{view}.{extension}",
            view=view,
        )

    log_activity(
        db,
        action="DOWNLOAD_REQUESTED",
        category="DOWNLOAD",
        details=f"Download requested for activity {activity_id}, view '{view}'.",
        severity="INFO",
        user=current_user,
    )
    return response
