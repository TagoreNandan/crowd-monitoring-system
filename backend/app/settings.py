"""Backend-only runtime settings.

This deliberately does NOT duplicate model/weights/device/risk/clustering
configuration -- that stays owned by ``crowd_monitoring.config`` and
``configs/default.yaml`` exactly as it is for the CLI scripts and the Flask
dashboard (see ``pipeline_runtime.build_pipeline_kwargs``). This class only
covers settings specific to running the FastAPI service itself, and every
value is environment-driven so nothing here hardcodes a production origin
or path.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CROWD_MONITORING_BACKEND_", case_sensitive=False, extra="ignore")

    environment: str = "development"

    # Comma-separated list of allowed CORS origins. Defaults to the two
    # ports Vite's dev server commonly uses; production origins must be
    # supplied via CROWD_MONITORING_BACKEND_CORS_ALLOW_ORIGINS, never hardcoded.
    cors_allow_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"

    log_level: str = "INFO"

    # Authentication & Initial Admin configuration
    jwt_secret_key: str = "cms-crowd-monitoring-secure-key-2026-default"
    jwt_expire_minutes: int = 10080  # 7 days
    admin_username: str = "admin"
    admin_password: str = "admin123"
    admin_email: str = "admin@cms.local"

    @model_validator(mode="after")
    def validate_production_secrets(self) -> Settings:
        is_dev = self.environment.lower() in ("development", "dev", "test", "testing")
        insecure_jwt = self.jwt_secret_key == "cms-crowd-monitoring-secure-key-2026-default" or len(self.jwt_secret_key) < 16
        insecure_admin = self.admin_password == "admin123"

        if not is_dev:
            if insecure_jwt:
                raise ValueError(
                    "In non-development environments, a secure, non-default CROWD_MONITORING_BACKEND_JWT_SECRET_KEY "
                    "(at least 16 characters) must be configured."
                )
            if insecure_admin:
                raise ValueError(
                    "In non-development environments, default admin_password ('admin123') is forbidden. "
                    "Set a strong CROWD_MONITORING_BACKEND_ADMIN_PASSWORD in your environment."
                )
        return self

    # How long an in-memory job record (and its temp working file, if any)
    # survives before being evicted. There is no persistent store in this
    # phase, so this is the only thing bounding memory/disk growth.
    job_ttl_seconds: float = 3600.0

    max_image_upload_mb: float = 25.0
    max_video_upload_mb: float = 2048.0

    # Confidence threshold for the shared person detector (visible_count/
    # detections). Weights are located via CROWD_MONITORING_YOLO_WEIGHTS (see
    # crowd_monitoring.detection) -- this only tunes how confident a detection must
    # be to count as a real person box.
    person_detection_confidence: float = 0.20

    # Detector inference resolution (long side). 1600 measured as the
    # latency/quality sweet spot for the CrowdHuman-trained detector on this
    # project's own 4K sample video frame and sample image (~83-84ms,
    # within ~5% of imgsz=1920's detection count); see crowd_monitoring.detection
    # module docstring for the full comparison.
    person_detection_imgsz: int = 1600

    # How often (times per second of SOURCE video) the expensive DM-Count
    # density/heatmap/localization/risk pass runs for video jobs and
    # real-time feeds. The person detector runs independently of this (every
    # decoded frame for video jobs; continuously against the latest frame
    # for feeds -- see crowd_monitoring.streaming.realtime) -- only the much slower
    # density pass is throttled, so boxes/tracking stay live while density/
    # heatmap/risk are refreshed periodically and reused in between. This is
    # what keeps a ~10s/609-frame 60fps video from taking minutes, and what
    # keeps a real-time feed from stalling while DM-Count runs: DM-Count
    # measured at ~500-550ms/frame on Apple MPS, so running it once per
    # source frame would either dominate total processing time (video) or
    # block live frame delivery entirely (feeds). ~1/sec is imperceptible
    # for a signal that doesn't change frame-to-frame anyway.
    density_analysis_hz: float = 1.0

    # How often (times per second of SOURCE video) the person detector runs
    # for a VIDEO JOB specifically (not images, not real-time feeds -- see
    # below). At ~85-180ms/frame (CrowdHuman YOLOv8n @ imgsz=1600, see
    # crowd_monitoring.detection), running the detector on every one of a 60fps
    # source's 609 frames is the single largest cost in processing the
    # project's own 4K sample video. 20/sec is still far smoother than
    # anything the human eye needs to perceive continuous motion (most video
    # reads as smooth well below that), while cutting detector calls to
    # roughly a third of "every frame". Frames between detector passes reuse
    # the tracker's last known boxes as-is (held, not re-detected) rather
    # than pretending a fresh detection happened -- `frames_detected` in the
    # job summary reports the real, honest count. Real-time feeds don't use
    # this setting: their fast path already runs the detector against
    # whatever frame is current as fast as it can (see
    # crowd_monitoring.streaming.realtime), which is a different, already-adaptive
    # cadence driven by actual detector throughput, not a fixed target.
    video_detection_hz: float = 20.0

    # JPEG quality (1-100) for annotated frames streamed over the real-time
    # feed WebSocket. Lower = smaller payload/less bandwidth, at the cost of
    # visual quality; 75 is a reasonable balance for a live preview.
    feed_frame_jpeg_quality: int = 75

    # Long side (px) annotated frames are downscaled to before JPEG-encoding
    # for the real-time feed WebSocket -- keeps a 4K source from sending a
    # multi-megabyte frame over the socket several times a second. Deliberately
    # independent of `person_detection_imgsz`: the detector always sees a
    # full/high-resolution frame regardless of what the browser is shown.
    feed_frame_max_side: int = 960

    # Nominal FPS for live-feed detection recordings when STORAGE_BACKEND=s3.
    # Heatmap and combined views are saved as timestamped JPEG snapshots.
    live_feed_recording_fps: float = 8.0

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
