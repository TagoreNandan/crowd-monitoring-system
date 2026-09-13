"""FastAPI application factory for the crowd-monitoring backend.

This wraps the existing `crowd_monitoring` package -- it does not duplicate any
inference, clustering, or risk logic. See docs in README.md ("FastAPI
backend") for the full endpoint list and the in-memory/temp-only storage
model this phase intentionally uses.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db.init_db import init_db
from .errors import install_exception_handlers
from .logging_config import configure_logging
from .pipeline_runtime import build_pipeline_kwargs
from .routers import admin as admin_router
from .routers import auth as auth_router
from .routers import config as config_router
from .routers import devices as devices_router
from .routers import feeds as feeds_router
from .routers import health as health_router
from .routers import jobs as jobs_router
from .routers import status as status_router
from .settings import get_settings
from .state import AppState
from .ws.router import router as ws_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Initialize SQLite database schema and seed default administrator
    try:
        init_db()
        logger.info("Database initialized successfully.")
    except Exception:
        logger.exception("Database initialization failed.")

    settings = get_settings()
    app_state = AppState(settings=settings, pipeline_kwargs=build_pipeline_kwargs())
    app.state.app_state = app_state

    try:
        await app_state.pipeline_runtime.ensure_ready()
        logger.info("crowd_monitoring pipeline ready.")
    except Exception:
        logger.exception("crowd_monitoring pipeline failed to load at startup; will retry lazily on first use.")

    yield

    app_state.pipeline_runtime.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Crowd Monitoring Platform API",
        description=(
            "FastAPI backend wrapping the crowd_monitoring AI pipeline (DM-Count density "
            "estimation, DBSCAN clustering, rule-based risk status)."
        ),
        version="0.1.0",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    install_exception_handlers(app)

    app.include_router(auth_router.router)
    app.include_router(admin_router.router)
    app.include_router(devices_router.router)
    app.include_router(health_router.router)
    app.include_router(status_router.router)
    app.include_router(config_router.router)
    app.include_router(jobs_router.router)
    app.include_router(feeds_router.router)
    app.include_router(ws_router)

    return app


app = create_app()
