"""Owns the single `crowdmon.pipeline.CrowdAnalysisPipeline` instance used by
the backend, and the thread-pool boundary that keeps its synchronous,
CPU/GPU-bound PyTorch inference off the asyncio event loop.

This module intentionally contains NO inference logic of its own -- it only
constructs and calls the existing `CrowdAnalysisPipeline` exactly as the CLI
scripts and Flask dashboard already do, via `crowdmon.config`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, TypeVar

import numpy as np

from crowdmon.config import load_config, pipeline_kwargs_from_config
from crowdmon.device import device_name
from crowdmon.pipeline import CrowdAnalysisPipeline, FrameResult

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def build_pipeline_kwargs() -> dict[str, Any]:
    """Resolve `CrowdAnalysisPipeline` kwargs the same way `dashboard/app.py`
    does: `configs/default.yaml` (or `CROWDMON_CONFIG`) as the base, with the
    same `CROWDMON_MODEL` / `CROWDMON_WEIGHTS` / `CROWDMON_DEVICE` env-var
    overrides. Keeping this identical to the dashboard means both front
    doors configure the underlying model the same way.
    """
    cfg = load_config(os.environ.get("CROWDMON_CONFIG"))
    kwargs = pipeline_kwargs_from_config(cfg)
    if os.environ.get("CROWDMON_MODEL"):
        kwargs["model_name"] = os.environ["CROWDMON_MODEL"]
    if os.environ.get("CROWDMON_WEIGHTS"):
        kwargs["model_weights"] = os.environ["CROWDMON_WEIGHTS"]
    if os.environ.get("CROWDMON_DEVICE"):
        kwargs["device"] = os.environ["CROWDMON_DEVICE"]
    return kwargs


class PipelineRuntime:
    """Lazily builds one `CrowdAnalysisPipeline` and runs all of its
    synchronous calls (construction + `.analyze()`) on a dedicated worker
    thread via `run_in_executor`, so an in-flight forward pass never blocks
    the event loop (and therefore never stalls WebSocket heartbeats or other
    requests).

    Model configuration is fixed at construction time -- there is no runtime
    hot-swap in this phase. `describe_config()` reports the pipeline's
    *actual resolved* state (e.g. the real device a "auto" request landed
    on) rather than echoing back the request, so it never claims something
    that isn't true yet.
    """

    def __init__(self, pipeline_kwargs: dict[str, Any], max_workers: int = 1):
        self._pipeline_kwargs = dict(pipeline_kwargs)
        self._pipeline: Optional[CrowdAnalysisPipeline] = None
        self._build_lock = threading.Lock()
        self._build_error: Optional[str] = None
        # A single density-regression forward pass gains nothing from >1
        # concurrent worker on one device, and >1 risks oversubscribing the
        # CPU -- one dedicated inference thread keeps requests serialized
        # and latency predictable.
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="crowdmon-inference")

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None

    @property
    def last_build_error(self) -> Optional[str]:
        return self._build_error

    def _build_pipeline_sync(self) -> CrowdAnalysisPipeline:
        """Runs on the worker thread. Safe to call repeatedly/concurrently:
        only the first caller actually builds the pipeline."""
        if self._pipeline is None:
            with self._build_lock:
                if self._pipeline is None:
                    logger.info("Building CrowdAnalysisPipeline with kwargs=%s", self._pipeline_kwargs)
                    try:
                        self._pipeline = CrowdAnalysisPipeline(**self._pipeline_kwargs)
                    except Exception as exc:
                        self._build_error = str(exc)
                        logger.exception("Failed to build CrowdAnalysisPipeline")
                        raise
        return self._pipeline

    async def ensure_ready(self) -> None:
        """Build the pipeline (downloads/loads weights as needed) off the
        event loop. Called once at startup; safe to call again later as a
        retry if the first attempt failed (e.g. offline at boot)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._build_pipeline_sync)

    async def get_pipeline(self) -> CrowdAnalysisPipeline:
        """Async helper to get or build the underlying CrowdAnalysisPipeline."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._build_pipeline_sync)

    async def analyze(self, frame_bgr: np.ndarray, growth_ratio: Optional[float] = None) -> FrameResult:
        """Async-safe wrapper around the existing, unmodified
        `CrowdAnalysisPipeline.analyze()`. `growth_ratio` is passed straight
        through to it -- it already accepts this for video's trend-based
        risk signal (see `crowdmon.video.process_source`); the video job
        runner supplies it, image jobs simply omit it."""
        loop = asyncio.get_running_loop()
        pipeline = await loop.run_in_executor(self._executor, self._build_pipeline_sync)
        return await loop.run_in_executor(
            self._executor, functools.partial(pipeline.analyze, frame_bgr, growth_ratio=growth_ratio)
        )

    async def run_sync(self, func: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
        """Run an arbitrary synchronous callable on the same dedicated
        worker thread used for inference. Used by the video job runner for
        OpenCV `VideoCapture` I/O so frame reads never touch the event loop
        either, and stay serialized with any concurrent inference work."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, functools.partial(func, *args, **kwargs))

    def describe_config(self) -> dict[str, Any]:
        """Real, current configuration -- resolved values if the pipeline is
        already built, otherwise the as-requested values with
        `device_resolved=False` so callers can't mistake "auto" for a real
        resolved device."""
        if self._pipeline is None:
            model_cfg = {
                "name": self._pipeline_kwargs.get("model_name"),
                "weights": self._pipeline_kwargs.get("model_weights"),
                "device": self._pipeline_kwargs.get("device", "auto"),
                "device_resolved": False,
                "max_side": self._pipeline_kwargs.get("max_side"),
            }
            risk_cfg = self._pipeline_kwargs.get("risk_config")
            risk = dataclasses.asdict(risk_cfg) if risk_cfg is not None else {}
        else:
            counter = self._pipeline.counter
            model_cfg = {
                "name": counter.model_name,
                "weights": counter.model_weights,
                "device": device_name(counter.device),
                "device_resolved": True,
                "max_side": counter.max_side,
            }
            risk = dataclasses.asdict(self._pipeline.risk_config)

        return {
            "model": model_cfg,
            "clustering": {
                "eps_px": self._pipeline_kwargs.get("eps_px"),
                "min_samples": self._pipeline_kwargs.get("min_samples"),
            },
            "risk": risk,
        }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
