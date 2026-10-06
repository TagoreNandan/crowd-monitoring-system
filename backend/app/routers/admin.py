"""Admin endpoints for user administration and system activity logs."""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc
from sqlalchemy.orm import Session

from ..auth.dependencies import log_activity, require_admin
from ..auth.security import hash_password
from ..db.models import ActivityLog, Device, User
from ..db.session import engine, get_db
from ..schemas.auth import (
    ActivityLogResponse,
    AdminOverviewStats,
    UserCreateRequest,
    UserResponse,
    UserUpdateRequest,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/stats", response_model=AdminOverviewStats)
def get_admin_stats(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Returns overview counts and the latest recent activity logs."""
    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()
    admin_users = db.query(User).filter(User.role == "admin").count()
    total_activities = db.query(ActivityLog).count()

    recent = (
        db.query(ActivityLog)
        .order_by(desc(ActivityLog.timestamp))
        .limit(10)
        .all()
    )

    return AdminOverviewStats(
        total_users=total_users,
        active_users=active_users,
        admin_users=admin_users,
        total_activities=total_activities,
        database_engine=engine.name,
        database_status="ONLINE",
        recent_activities=recent,
    )


@router.get("/users", response_model=List[UserResponse])
def list_users(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin-only: list all registered user accounts."""
    return db.query(User).order_by(desc(User.created_at)).all()


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    req: UserCreateRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin-only: create a new user account with specified role."""
    existing = db.query(User).filter(
        (User.username == req.username) | (User.email == req.email)
    ).first()
    if existing:
        if existing.username == req.username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username is already registered",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is already registered",
        )

    role = req.role if req.role in ("admin", "analyst") else "analyst"
    user = User(
        username=req.username,
        email=req.email,
        full_name=req.full_name or "",
        hashed_password=hash_password(req.password),
        role=role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    log_activity(
        db,
        action="USER_CREATED_BY_ADMIN",
        category="SECURITY",
        details=f"Admin {admin.username} created user {user.username} with role {user.role}",
        severity="WARNING" if user.role == "admin" else "INFO",
        user=admin,
    )

    return user


@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin-only: remove/delete a user account."""
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if target_user.id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own administrator account",
        )

    # Detach any devices created by this user
    db.query(Device).filter(Device.created_by == target_user.id).update({Device.created_by: None})

    username = target_user.username
    db.delete(target_user)
    db.commit()

    log_activity(
        db,
        action="USER_DELETED_BY_ADMIN",
        category="SECURITY",
        details=f"Admin {admin.username} deleted user account {username}",
        severity="WARNING",
        user=admin,
    )

    return {"status": "success", "message": f"User {username} deleted"}


@router.patch("/users/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    req: UserUpdateRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin-only: update user role or deactivate/activate account."""
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Prevent admin from deactivating themselves
    if target_user.id == admin.id and req.is_active is False:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own administrator account",
        )

    if req.is_active is not None:
        target_user.is_active = bool(req.is_active)  # type: ignore[assignment]
    if req.role is not None:
        normalized_role = "analyst" if req.role == "user" else req.role
        if normalized_role not in ("admin", "analyst"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Role must be either 'admin' or 'analyst'",
            )
        target_user.role = str(normalized_role)  # type: ignore[assignment]
    if req.full_name is not None:
        target_user.full_name = str(req.full_name)  # type: ignore[assignment]

    db.commit()
    db.refresh(target_user)

    log_activity(
        db,
        action="USER_UPDATED_BY_ADMIN",
        category="SECURITY",
        details=f"Admin {admin.username} updated user {target_user.username} (role={target_user.role}, active={target_user.is_active})",
        severity="WARNING",
        user=admin,
    )

    return target_user


@router.get("/activities", response_model=List[ActivityLogResponse])
def get_activities(
    limit: int = Query(50, ge=1, le=500),
    category: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin-only: query chronological system activity log entries."""
    query = db.query(ActivityLog)
    if category:
        query = query.filter(ActivityLog.category == category)
    if severity:
        query = query.filter(ActivityLog.severity == severity)

    return query.order_by(desc(ActivityLog.timestamp)).limit(limit).all()
