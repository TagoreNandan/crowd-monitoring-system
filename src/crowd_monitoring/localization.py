"""Approximate point localization derived from the density map.

DM-Count / CSRNet are density-regression models: they were trained to
predict a smooth per-pixel density field, not discrete head coordinates,
so they do not ship a dedicated localization head the way a detector
(FIDTM's FIDT map, P2PNet, a head-detector) would.

Rather than inventing fake head positions, we extract *local maxima* of the
predicted density map -- a standard, well understood technique (it's the
same idea FIDTM itself uses to turn its FIDT map into discrete points) --
and treat the result as an approximate set of "concentration points" for
downstream spatial analysis (clustering, hotspot detection).

Important, documented limitation: in a sparse-to-medium crowd, each peak
usually corresponds to roughly one person and the peak count tracks the
model's integrated count reasonably well. In a very dense, heavily
occluded crowd the individual density bumps merge together, so the number
of resolvable peaks under-counts the true number of people -- the
*integrated* density (`InferenceResult.count`) remains the reliable count
signal in that regime, not the number of peaks. Do not treat the peaks as
a precise per-person detector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter


@dataclass
class LocalizationResult:
    points_xy: np.ndarray  # Nx2 float array, in ORIGINAL image pixel coordinates
    scores: np.ndarray  # N float array, density value at each peak
    method: str = "density-map local maxima (approximate)"


def extract_points(
    density_map: np.ndarray,
    original_size: tuple,
    min_distance: int = 3,
    threshold_rel: float = 0.10,
) -> LocalizationResult:
    """Find local density maxima and rescale them into original image coordinates.

    min_distance: neighborhood radius (in density-map pixels) used for
        non-max suppression -- larger values merge nearby peaks.
    threshold_rel: peaks below `threshold_rel * max(density_map)` are
        discarded as noise.
    """
    d = density_map.astype(np.float32)
    if d.max() <= 0:
        return LocalizationResult(points_xy=np.zeros((0, 2)), scores=np.zeros((0,)))

    footprint_size = 2 * min_distance + 1
    local_max = maximum_filter(d, size=footprint_size) == d
    threshold = d.max() * threshold_rel
    mask = local_max & (d > threshold)

    ys, xs = np.nonzero(mask)
    scores = d[ys, xs]

    dh, dw = d.shape
    oh, ow = original_size
    scale_x = ow / dw
    scale_y = oh / dh
    points_xy = np.stack([xs * scale_x, ys * scale_y], axis=1)

    return LocalizationResult(points_xy=points_xy, scores=scores)
