"""End-to-end orchestration for a single frame:

    frame -> model inference -> density map -> heatmap/overlay
          -> approximate localization points -> DBSCAN clusters
          -> rule-based risk/status

Used identically by the image CLI, the video/webcam loop, and the
dashboard backend so all three stay in sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .clustering import ClusterStats, cluster_points
from .heatmap import colorize_density, overlay_heatmap
from .inference import CrowdCounter, InferenceResult
from .localization import LocalizationResult, extract_points
from .risk import RiskAssessment, RiskConfig, assess_risk
from .system_status import GLOBAL_STATUS


@dataclass
class FrameResult:
    inference: InferenceResult
    localization: LocalizationResult
    clusters: ClusterStats
    risk: RiskAssessment
    heatmap_bgr: np.ndarray
    overlay_bgr: np.ndarray


class CrowdAnalysisPipeline:
    def __init__(
        self,
        model_name: str = "DM-Count",
        model_weights: str = "QNRF",
        device: str = "auto",
        max_side: int = 1024,
        eps_px: float = 40.0,
        min_samples: int = 3,
        risk_config: Optional[RiskConfig] = None,
        download: bool = True,
    ):
        self.counter = CrowdCounter(
            model_name=model_name, model_weights=model_weights, device=device, max_side=max_side, download=download
        )
        self.eps_px = eps_px
        self.min_samples = min_samples
        self.risk_config = risk_config or RiskConfig()
        GLOBAL_STATUS.model_ready(model_name, model_weights, self.counter.device.type)

    def analyze(self, frame_bgr: np.ndarray, resize: bool = True, growth_ratio: Optional[float] = None) -> FrameResult:
        try:
            inference = self.counter.infer(frame_bgr, resize=resize)
        except Exception as exc:
            GLOBAL_STATUS.record_error(str(exc))
            raise
        GLOBAL_STATUS.record_inference(inference.latency_ms)

        loc = extract_points(inference.density_map, inference.original_size)
        clusters = cluster_points(loc.points_xy, eps_px=self.eps_px, min_samples=self.min_samples)

        h, w = inference.original_size
        risk = assess_risk(
            count=inference.count,
            frame_area_px=float(h * w),
            largest_cluster_size=clusters.largest_cluster_size,
            n_points=len(loc.points_xy),
            growth_ratio=growth_ratio,
            config=self.risk_config,
        )

        heatmap_bgr = colorize_density(inference.density_map, output_size=inference.original_size)
        overlay_bgr = overlay_heatmap(frame_bgr, inference.density_map)

        return FrameResult(
            inference=inference,
            localization=loc,
            clusters=clusters,
            risk=risk,
            heatmap_bgr=heatmap_bgr,
            overlay_bgr=overlay_bgr,
        )
