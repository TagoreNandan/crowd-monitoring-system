"""Database initialization and initial administrator account seeding."""

from __future__ import annotations

import logging

try:
    from .session import Base, SessionLocal, engine
    from .models import User, ActivityLog, Device
    from ..auth.security import hash_password
    from ..settings import get_settings
except (ImportError, ValueError):
    from backend.app.db.session import Base, SessionLocal, engine
    from backend.app.db.models import User, ActivityLog, Device
    from backend.app.auth.security import hash_password
    from backend.app.settings import get_settings

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

        # Seed initial sample camera if fleet is empty
        if db.query(Device).count() == 0:
            sample_device = Device(
                device_id="CAM-DEMO-01",
                name="North Concourse Main Gate",
                description="High-definition 4K sample surveillance feed for crowd monitoring.",
                location="Main Terminal Gate 3",
                stream_type="VIDEO_FILE",
                stream_url="examples/sample_crowd.mp4",
                status="ONLINE",
                fps=60.0,
                resolution="3840x2160",
                created_by=admin.id if admin else None,
            )
            db.add(sample_device)
            db.commit()
            logger.info("Default demonstration device seeded: CAM-DEMO-01")

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    logging.basicConfig(level=logging.INFO)
    init_db()
    print("Database initialization completed successfully.")

