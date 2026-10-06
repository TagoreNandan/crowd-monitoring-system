"""Database initialization and initial administrator account seeding."""

from __future__ import annotations

import logging

from sqlalchemy import inspect, text

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


def _ensure_activity_log_columns() -> None:
    """Add activity output-reference columns to databases created by older versions."""
    required_columns = {
        "job_id": "VARCHAR",
        "job_type": "VARCHAR",
        "output_keys": "TEXT",
        "storage_backend": "VARCHAR",
    }
    inspector = inspect(engine)
    existing_columns = {column["name"] for column in inspector.get_columns("activity_logs")}
    with engine.begin() as connection:
        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                connection.execute(text(f"ALTER TABLE activity_logs ADD COLUMN {column_name} {column_type}"))
        if "job_id" not in {index["name"] for index in inspector.get_indexes("activity_logs")}:
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_activity_logs_job_id ON activity_logs (job_id)"))


def init_db() -> None:
    """Create database tables and ensure administrator account exists."""
    Base.metadata.create_all(bind=engine)
    _ensure_activity_log_columns()

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
            logger.info(f"Administrator account '{settings.admin_username}' already exists; preserving existing credentials.")

        # Seed default demonstration analyst account if not present (only in development environments)
        if settings.environment.lower() in ("development", "dev", "test", "testing"):
            analyst = db.query(User).filter(User.username == "analyst").first()
            if not analyst:
                analyst = User(
                    username="analyst",
                    email="analyst@cms.local",
                    full_name="Operations Analyst",
                    hashed_password=hash_password("analyst123"),
                    role="analyst",
                    is_active=True,
                )
                db.add(analyst)
                db.commit()
                logger.info("Default demonstration analyst account seeded: username='analyst'")

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

