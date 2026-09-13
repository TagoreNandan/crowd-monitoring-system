"""Database initialization and initial administrator account seeding."""

from __future__ import annotations

import logging
from .session import Base, SessionLocal, engine
from .models import User, ActivityLog
from ..auth.security import hash_password
from ..settings import get_settings

logger = logging.getLogger(__name__)


def init_db() -> None:
    """Create database tables and ensure administrator account exists."""
    Base.metadata.create_all(bind=engine)

    settings = get_settings()
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == settings.admin_username).first()
        if not admin:
            admin = User(
                username=settings.admin_username,
                email=settings.admin_email,
                full_name="System Administrator",
                hashed_password=hash_password(settings.admin_password),
                role="admin",
                is_active=True,
            )
            db.add(admin)
            db.commit()
            db.refresh(admin)
            logger.info(f"Administrator account seeded: username='{settings.admin_username}'")

            log_entry = ActivityLog(
                user_id=admin.id,
                username=settings.admin_username,
                action="SYSTEM_INIT",
                category="SYSTEM",
                details="Initial system database setup and administrator account configured.",
                severity="INFO",
            )
            db.add(log_entry)
            db.commit()
        else:
            # Reassert password and active admin status
            admin.hashed_password = hash_password(settings.admin_password)
            admin.role = "admin"
            admin.is_active = True
            db.commit()
            logger.info(f"Administrator credentials synced: username='{settings.admin_username}'")

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()
