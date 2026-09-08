"""Model loading and single-frame crowd-counting inference.

Wraps a density-regression backbone (DM-Count by default, CSRNet optional)
behind a small, dependency-light API: give it a BGR numpy image (as
produced by cv2.imread / cv2.VideoCapture), get back an estimated count and
the raw density map the count was integrated from.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .device import device_name, get_device_lock, resolve_device
from .models import build_model

logger = logging.getLogger(__name__)


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class InferenceResult:
    count: float
    density_map: np.ndarray  # HxW, float32, same content-scale as the resized input
    input_size: tuple  # (h, w) actually fed to the model
    original_size: tuple  # (h, w) of the frame passed in
    latency_ms: float
    model_name: str
    model_weights: str
    device: str
    warnings: list = field(default_factory=list)


class CrowdCounter:
    """Loads a pretrained density-regression model once and runs inference on frames."""

    def __init__(
        self,
        model_name: str = "DM-Count",
        model_weights: str = "QNRF",
        device: str = "auto",
        max_side: int = 1024,
        download: bool = True,
    ):
        self.model_name = model_name
        self.model_weights = model_weights
        self.max_side = max_side
        self.device = resolve_device(device)
        self._warnings: list[str] = []

        logger.info("Loading %s (%s) on %s", model_name, model_weights, device_name(self.device))
        try:
            self.model = build_model(model_name, model_weights, self.device, download=download)
        except RuntimeError as exc:
            # e.g. an unsupported op on MPS in this torch build -- fall back to CPU rather than crash.
            if self.device.type != "cpu":
                logger.warning("Model failed to run on %s (%s); falling back to CPU.", self.device, exc)
                self._warnings.append(f"{self.device.type} backend failed ({exc}); using CPU instead.")
                self.device = torch.device("cpu")
                self.model = build_model(model_name, model_weights, self.device, download=download)
            else:
                raise

    def _preprocess(self, frame_bgr: np.ndarray, resize: bool) -> tuple[torch.Tensor, tuple]:
        h, w = frame_bgr.shape[:2]
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if resize:
            long_side = max(h, w)
            if long_side > self.max_side:
                factor = self.max_side / long_side
                new_w, new_h = int(round(w * factor)), int(round(h * factor))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor, img.shape[:2]

    def infer(self, frame_bgr: np.ndarray, resize: bool = True) -> InferenceResult:
        """Run one forward pass and return the estimated count + density map."""
        t0 = time.perf_counter()
        original_size = frame_bgr.shape[:2]
        tensor, input_size = self._preprocess(frame_bgr, resize)
        tensor = tensor.to(self.device)

        try:
            with get_device_lock(self.device):
                with torch.no_grad():
                    output = self.model(tensor)
        except RuntimeError as exc:
            if self.device.type != "cpu":
                logger.warning("Inference failed on %s (%s); retrying on CPU.", self.device, exc)
                self._warnings.append(f"{self.device.type} inference failed mid-run ({exc}); switched to CPU.")
                self.device = torch.device("cpu")
                self.model.to(self.device)
                tensor = tensor.to(self.device)
                with get_device_lock(self.device):
                    with torch.no_grad():
                        output = self.model(tensor)
            else:
                raise

        density_map = output[0, 0].detach().cpu().numpy().astype(np.float32)
        count = float(density_map.sum())
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return InferenceResult(
            count=count,
            density_map=density_map,
            input_size=tuple(input_size),
            original_size=tuple(original_size),
            latency_ms=latency_ms,
            model_name=self.model_name,
            model_weights=self.model_weights,
            device=device_name(self.device),
            warnings=list(self._warnings),
        )
