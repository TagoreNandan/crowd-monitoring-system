"""Hardware backend selection: CUDA > Apple MPS > CPU.

No backend is hardcoded. The pipeline asks for "auto" by default and this
module probes what the current machine actually supports, so the same code
runs unchanged on an Apple Silicon laptop, a CUDA workstation, or a plain
CPU-only container like this one.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


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
