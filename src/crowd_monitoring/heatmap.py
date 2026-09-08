"""Turn a raw density map into visualizations: a colorized heatmap and an
overlay blended onto the original frame.

Everything here is derived from the model's own per-image output -- there
are no static/precomputed images anywhere in this module.
"""

from __future__ import annotations

import cv2
import numpy as np


def normalize_density(density_map: np.ndarray) -> np.ndarray:
    """Min-max normalize a density map to [0, 1] for visualization only.

    (The un-normalized map is what gets summed for the count; this is a
    separate, display-only rescaling.)
    """
    d = density_map.astype(np.float32)
    lo, hi = float(d.min()), float(d.max())
    if hi - lo < 1e-12:
        return np.zeros_like(d)
    return (d - lo) / (hi - lo)


def colorize_density(density_map: np.ndarray, output_size: tuple | None = None, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Render a density map as a BGR heatmap image, resized to `output_size` (h, w) if given."""
    norm = normalize_density(density_map)
    gray = (norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(gray, colormap)
    if output_size is not None:
        h, w = output_size
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    return heatmap


def overlay_heatmap(frame_bgr: np.ndarray, density_map: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend a colorized density heatmap on top of the original frame."""
    h, w = frame_bgr.shape[:2]
    heatmap = colorize_density(density_map, output_size=(h, w))
    return cv2.addWeighted(heatmap, alpha, frame_bgr, 1 - alpha, 0)
