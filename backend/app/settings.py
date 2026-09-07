"""Backend-only runtime settings.

This deliberately does NOT duplicate model/weights/device/risk/clustering
configuration -- that stays owned by ``crowdmon.config`` and
``configs/default.yaml`` exactly as it is for the CLI scripts and the Flask
dashboard (see ``pipeline_runtime.build_pipeline_kwargs``). This class only
covers settings specific to running the FastAPI service itself, and every
value is environment-driven so nothing here hardcodes a production origin
or path.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CROWDMON_BACKEND_", case_sensitive=False, extra="ignore")

    environment: str = "development"

    # Comma-separated list of allowed CORS origins. Defaults to the two
    # ports Vite's dev server commonly uses; production origins must be
    # supplied via CROWDMON_BACKEND_CORS_ALLOW_ORIGINS, never hardcoded.
    cors_allow_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    log_level: str = "INFO"

    # How long an in-memory job record (and its temp working file, if any)
    # survives before being evicted. There is no persistent store in this
    # phase, so this is the only thing bounding memory/disk growth.
    job_ttl_seconds: float = 3600.0

    max_image_upload_mb: float = 25.0
    max_video_upload_mb: float = 2048.0

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
