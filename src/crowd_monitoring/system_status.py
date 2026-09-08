"""Backend/system status tracking for the dashboard.

Tracks real, locally-observed state only (model loaded, source connected,
last inference time/latency, FPS, error count). Never claims a cloud
connection -- everything here runs in this process.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SystemStatus:
    model_loaded: bool = False
    model_name: str = ""
    model_weights: str = ""
    device: str = ""
    source_connected: bool = False
    source_description: str = ""
    inference_running: bool = False
    last_inference_ts: Optional[float] = None
    last_latency_ms: Optional[float] = None
    fps: float = 0.0
    frames_processed: int = 0
    error_count: int = 0
    last_error: str = ""

    def state(self, stale_after_s: float = 15.0) -> str:
        """ONLINE / DEGRADED / OFFLINE summary."""
        if not self.model_loaded:
            return "OFFLINE"
        if self.error_count > 0 and self.last_error:
            recent_error = self.last_inference_ts is None or (time.time() - self.last_inference_ts) < stale_after_s
            if recent_error:
                return "DEGRADED"
        if self.inference_running and self.last_inference_ts is not None:
            if (time.time() - self.last_inference_ts) > stale_after_s:
                return "DEGRADED"
        return "ONLINE"

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["state"] = self.state()
        return d


class StatusTracker:
    """Thread-safe holder for SystemStatus, updated as the pipeline runs."""

    def __init__(self):
        self._status = SystemStatus()
        self._lock = threading.Lock()
        self._frame_times: list[float] = []

    def model_ready(self, model_name: str, model_weights: str, device: str):
        with self._lock:
            self._status.model_loaded = True
            self._status.model_name = model_name
            self._status.model_weights = model_weights
            self._status.device = device

    def source_ready(self, description: str):
        with self._lock:
            self._status.source_connected = True
            self._status.source_description = description

    def source_closed(self):
        with self._lock:
            self._status.source_connected = False

    def record_inference(self, latency_ms: float):
        with self._lock:
            now = time.time()
            self._status.inference_running = True
            self._status.last_inference_ts = now
            self._status.last_latency_ms = latency_ms
            self._status.frames_processed += 1

            self._frame_times.append(now)
            cutoff = now - 5.0
            self._frame_times = [t for t in self._frame_times if t >= cutoff]
            if len(self._frame_times) >= 2:
                span = self._frame_times[-1] - self._frame_times[0]
                self._status.fps = (len(self._frame_times) - 1) / span if span > 0 else 0.0
            else:
                self._status.fps = 0.0

    def record_error(self, message: str):
        with self._lock:
            self._status.error_count += 1
            self._status.last_error = message

    def stop_inference(self):
        with self._lock:
            self._status.inference_running = False

    def snapshot(self) -> SystemStatus:
        with self._lock:
            return SystemStatus(**self._status.__dict__)


# A single process-wide tracker used by the CLI scripts and the dashboard.
GLOBAL_STATUS = StatusTracker()
