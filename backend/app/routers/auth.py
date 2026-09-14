"""Authentication and User Registration endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..auth.dependencies import get_current_user, log_activity
from ..auth.security import create_access_token, hash_password, verify_password
from ..db.models import User
from ..db.session import get_db
from ..schemas.auth import TokenResponse, UserLoginRequest, UserRegisterRequest, UserResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(
    req: UserRegisterRequest,
    db: Session = Depends(get_db),
):
    """Register a new user account."""
    # Check if username or email already taken
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

    user = User(
        username=req.username,
        email=req.email,
        full_name=req.full_name or "",
        hashed_password=hash_password(req.password),
        role="analyst",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    log_activity(
        db,
        action="USER_REGISTERED",
        category="AUTH",
        details=f"New user account registered: {user.username} ({user.email})",
        severity="INFO",
        user=user,
    )

    return user


@router.post("/login", response_model=TokenResponse)
def login(
    req: UserLoginRequest,
    db: Session = Depends(get_db),
):
    """Log in with username or email and password, returning a JWT token."""
    username_clean = (req.username or "").strip()
    password_clean = (req.password or "").strip()

    user = (
        db.query(User)
        .filter(
            (func.lower(User.username) == username_clean.lower())
            | (func.lower(User.email) == username_clean.lower())
        )
        .first()
    )

    if not user or not verify_password(password_clean, user.hashed_password):
        log_activity(
            db,
            action="LOGIN_FAILED",
            category="AUTH",
            details=f"Failed login attempt for identifier: {username_clean}",
            severity="WARNING",
            username=username_clean or "unknown",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Contact an administrator.",
        )

    user.last_login = datetime.now(timezone.utc)
    db.commit()

    token = create_access_token(
        data={"sub": user.username, "role": user.role, "id": user.id}
    )

    log_activity(
        db,
        action="USER_LOGIN",
        category="AUTH",
        details=f"User {user.username} logged in successfully.",
        severity="INFO",
        user=user,
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        username=user.username,
        role=user.role,
        full_name=user.full_name or user.username,
    )


@router.post("/token", response_model=TokenResponse)
def login_oauth2(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """Standard OAuth2 form-compatible endpoint."""
    req = UserLoginRequest(username=form_data.username, password=form_data.password)
    return login(req=req, db=db)


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Return the profile of the currently authenticated user."""
    return current_user
