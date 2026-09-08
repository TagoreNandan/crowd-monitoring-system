"""Ground-truth-based evaluation: MAE / RMSE / PSNR / SSIM.

Why this is a separate, opt-in mode
------------------------------------
PSNR and SSIM are only meaningful when you have two things of the same
kind to compare -- e.g. a predicted density map against a *ground-truth*
density map built from real head annotations for that exact image. There
is no ground truth for an arbitrary photo a user drops into the demo, so
computing PSNR/SSIM during normal inference would produce numbers that
*look* like the reference screenshots but mean nothing (there's nothing
correct to compare against). Academic crowd-counting papers (including
FIDTM) only report PSNR/SSIM on benchmark datasets (ShanghaiTech, UCF-QNRF,
...) where ground-truth density maps exist.

This module is used only when the caller explicitly supplies ground truth,
either as:
  * a ground-truth density map (numpy array / .npy), or
  * a set of ground-truth head-point (x, y) annotations, from which a
    density map is built the standard way (a Gaussian placed at each
    point).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass
class EvaluationResult:
    mae: float  # |pred_count - gt_count|
    rmse: float  # sqrt((pred_count - gt_count)^2)  -- same as MAE for a single image, kept for API symmetry
    psnr: float | None
    ssim: float | None
    pred_count: float
    gt_count: float
    note: str = ""


def build_gt_density_map(points_xy: np.ndarray, image_size: tuple, sigma: float = 15.0) -> np.ndarray:
    """Build a ground-truth density map from head-point annotations.

    Standard construction used across the crowd-counting literature: place
    a delta at each annotated point and blur with a fixed-sigma Gaussian
    so the map integrates to the annotated count.
    """
    h, w = image_size
    density = np.zeros((h, w), dtype=np.float32)
    for x, y in points_xy:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < h and 0 <= xi < w:
            density[yi, xi] += 1.0
    if len(points_xy) > 0:
        density = gaussian_filter(density, sigma=sigma, mode="constant")
    return density


def _resize_to_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize the larger map down to the smaller map's shape (nearest-safe, area-preserving)."""
    import cv2

    if a.shape == b.shape:
        return a, b
    target_h = min(a.shape[0], b.shape[0])
    target_w = min(a.shape[1], b.shape[1])

    def _resize(arr):
        if arr.shape[:2] == (target_h, target_w):
            return arr
        scale = (target_h * target_w) / (arr.shape[0] * arr.shape[1])
        resized = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return resized / scale  # preserve the integral (=count) under resizing

    return _resize(a), _resize(b)


def evaluate_density_maps(pred_density: np.ndarray, gt_density: np.ndarray) -> EvaluationResult:
    """Compare a predicted density map against a ground-truth density map."""
    pred_count = float(pred_density.sum())
    gt_count = float(gt_density.sum())
    mae = abs(pred_count - gt_count)
    rmse = mae  # single-sample RMSE degenerates to MAE; kept distinct for multi-image extensions

    p, g = _resize_to_match(pred_density.astype(np.float32), gt_density.astype(np.float32))

    data_range = float(max(p.max(), g.max()) - min(p.min(), g.min()))
    if data_range < 1e-9:
        psnr = None
        ssim = None
        note = "Density maps are (near-)constant; PSNR/SSIM are undefined."
    else:
        psnr = float(peak_signal_noise_ratio(g, p, data_range=data_range))
        min_dim = min(p.shape)
        if min_dim < 3:
            ssim = None
            note = "Density maps are too small (<3px on one side) for a meaningful SSIM window; PSNR only."
        else:
            # structural_similarity requires win_size odd and <= the smallest map dimension.
            win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
            ssim = float(structural_similarity(g, p, data_range=data_range, win_size=win_size))
            note = "Computed against a supplied ground-truth density map."

    return EvaluationResult(mae=mae, rmse=rmse, psnr=psnr, ssim=ssim, pred_count=pred_count, gt_count=gt_count, note=note)


def evaluate_with_gt_points(pred_density: np.ndarray, gt_points_xy: np.ndarray, image_size: tuple, sigma: float = 15.0) -> EvaluationResult:
    """Convenience wrapper: build a GT density map from point annotations, then evaluate."""
    gt_density = build_gt_density_map(gt_points_xy, image_size, sigma=sigma)
    return evaluate_density_maps(pred_density, gt_density)


def evaluate_count_only(pred_count: float, gt_count: float) -> EvaluationResult:
    """When only a ground-truth *count* (not a map or points) is available: MAE/RMSE only, no PSNR/SSIM."""
    mae = abs(pred_count - gt_count)
    return EvaluationResult(
        mae=mae,
        rmse=mae,
        psnr=None,
        ssim=None,
        pred_count=pred_count,
        gt_count=gt_count,
        note="Only a ground-truth count was supplied; PSNR/SSIM require a ground-truth map or point annotations.",
    )
