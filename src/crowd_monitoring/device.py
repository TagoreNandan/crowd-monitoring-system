"""Hardware backend selection: CUDA > Apple MPS > CPU.

No backend is hardcoded. The pipeline asks for "auto" by default and this
module probes what the current machine actually supports, so the same code
runs unchanged on an Apple Silicon laptop, a CUDA workstation, or a plain
CPU-only container like this one.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_MPS_LOCK = threading.Lock()
_NULL_CONTEXT = contextlib.nullcontext()


def get_device_lock(device: Optional[torch.device] = None):
    """Serialization guard for a torch forward pass.

    Apple's MPS backend is not safe for truly concurrent kernel launches
    from multiple Python threads on the torch versions this project has
    been tested against -- calling into it from two threads at once can
    crash the process. This project runs two independent model-execution
    threads for a real-time feed (the fast person-detector loop and the
    slow DM-Count loop, see `crowd_monitoring.streaming.realtime`), so *some*
    serialization is required whenever either one is on MPS.

    It is scoped to MPS specifically (not "any device") on purpose: a CPU
    forward pass from one thread and an MPS forward pass from another don't
    share any hardware kernel queue, so forcing them through the same lock
    would serialize work that doesn't need to be serialized -- exactly the
    kind of unnecessary blocking that made an earlier version of the
    real-time feed stall while DM-Count ran. Pass `device=None` (or a
    non-MPS device) to get a no-op context manager.
    """
    if device is not None and device.type == "mps":
        return _MPS_LOCK
    return _NULL_CONTEXT


def resolve_device(preferred: str = "auto") -> torch.device:
    """Resolve a torch.device from a user preference.

    preferred: "auto", "cpu", "cuda", or "mps".
    Falls back to CPU with a warning if the requested backend isn't
    available on this machine.
    """
    preferred = (preferred or "auto").lower()

    if preferred == "cpu":
        return torch.device("cpu")

    if preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")

    if preferred == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("MPS requested but not available; falling back to CPU.")
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        try:
            return f"cuda ({torch.cuda.get_device_name(device)})"
        except Exception:
            return "cuda"
    if device.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"
