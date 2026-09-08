"""Rule-based crowd risk / status engine.

This is explicitly NOT a validated stampede-prediction model. It is a
transparent, tunable set of heuristics that turns live inference signals
(count, spatial density, cluster concentration, and -- for video -- how
fast the count is changing) into an easy-to-read status: SAFE, CAUTION,
WARNING, or CRITICAL. All thresholds live in one place (RiskConfig) and
are meant to be calibrated per deployment (camera height/angle, venue
capacity, etc.), not treated as universal truths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

STATUS_LEVELS = ["SAFE", "CAUTION", "WARNING", "CRITICAL"]


@dataclass
class RiskConfig:
    # Absolute count thresholds (people in frame) -- tune to venue/camera FOV.
    count_caution: float = 30
    count_warning: float = 80
    count_critical: float = 150

    # Relative density: estimated people per 100x100 px patch of frame area.
    # A rough, unitless proxy for "how packed" the crowd looks -- NOT a
    # calibrated people-per-square-meter figure, since that requires known
    # camera geometry which a single arbitrary image doesn't provide.
    density_caution: float = 0.6
    density_warning: float = 1.3
    density_critical: float = 2.2

    # Fraction of all localization points that fall inside the single
    # largest DBSCAN cluster -- a high value means the crowd is
    # concentrated in one spot rather than spread out.
    concentration_caution: float = 0.35
    concentration_warning: float = 0.55
    concentration_critical: float = 0.75

    # Relative growth in count over the trend window (video only), e.g.
    # 0.20 = count grew >=20% since the start of the window.
    growth_caution: float = 0.15
    growth_warning: float = 0.35
    growth_critical: float = 0.60

    # Blend weights for the four component scores (auto-normalized).
    weight_count: float = 0.35
    weight_density: float = 0.25
    weight_concentration: float = 0.20
    weight_growth: float = 0.20


@dataclass
class RiskAssessment:
    status: str
    score: float  # 0-100
    components: dict = field(default_factory=dict)
    explanation: str = ""


def _score_against_thresholds(value: float, caution: float, warning: float, critical: float) -> float:
    """Piecewise-linear map from a raw metric to a 0-100 sub-score."""
    if value <= 0:
        return 0.0
    if value < caution:
        return 25.0 * (value / caution)
    if value < warning:
        return 25.0 + 25.0 * (value - caution) / max(warning - caution, 1e-9)
    if value < critical:
        return 50.0 + 25.0 * (value - warning) / max(critical - warning, 1e-9)
    # Beyond "critical" the score keeps climbing slowly so a wildly
    # over-threshold reading still shows as worse than a barely-critical one.
    extra = min((value - critical) / max(critical, 1e-9), 1.0)
    return 75.0 + 25.0 * extra


def assess_risk(
    count: float,
    frame_area_px: float,
    largest_cluster_size: int,
    n_points: int,
    growth_ratio: Optional[float] = None,
    config: RiskConfig = None,
) -> RiskAssessment:
    """Compute a risk score/status from live inference signals.

    growth_ratio: (current_count - count_at_start_of_window) / count_at_start_of_window,
        or None if there isn't enough history yet (single image, or first
        few frames of a video) -- in that case the growth component is
        treated as neutral (0) rather than penalizing.
    """
    cfg = config or RiskConfig()

    density = count / max(frame_area_px, 1.0) * (100 * 100)
    concentration = (largest_cluster_size / n_points) if n_points > 0 else 0.0
    growth = growth_ratio if growth_ratio is not None else 0.0

    count_score = _score_against_thresholds(count, cfg.count_caution, cfg.count_warning, cfg.count_critical)
    density_score = _score_against_thresholds(density, cfg.density_caution, cfg.density_warning, cfg.density_critical)
    concentration_score = _score_against_thresholds(
        concentration, cfg.concentration_caution, cfg.concentration_warning, cfg.concentration_critical
    )
    growth_score = _score_against_thresholds(
        max(growth, 0.0), cfg.growth_caution, cfg.growth_warning, cfg.growth_critical
    ) if growth_ratio is not None else 0.0

    total_weight = cfg.weight_count + cfg.weight_density + cfg.weight_concentration + cfg.weight_growth
    score = (
        cfg.weight_count * count_score
        + cfg.weight_density * density_score
        + cfg.weight_concentration * concentration_score
        + cfg.weight_growth * growth_score
    ) / max(total_weight, 1e-9)

    if score < 25:
        status = "SAFE"
    elif score < 50:
        status = "CAUTION"
    elif score < 75:
        status = "WARNING"
    else:
        status = "CRITICAL"

    components = {
        "count": round(count, 1),
        "count_score": round(count_score, 1),
        "relative_density_per_100px2": round(density, 3),
        "density_score": round(density_score, 1),
        "cluster_concentration": round(concentration, 3),
        "concentration_score": round(concentration_score, 1),
        "growth_ratio": None if growth_ratio is None else round(growth_ratio, 3),
        "growth_score": round(growth_score, 1),
    }

    reasons = []
    if count_score >= 50:
        reasons.append(f"count is high ({count:.0f})")
    if density_score >= 50:
        reasons.append("frame is densely packed")
    if concentration_score >= 50:
        reasons.append("crowd is concentrated in one cluster")
    if growth_ratio is not None and growth_score >= 50:
        reasons.append(f"count is rising quickly ({growth_ratio * 100:.0f}% over window)")
    explanation = "; ".join(reasons) if reasons else "no elevated risk signals"

    return RiskAssessment(status=status, score=round(score, 1), components=components, explanation=explanation)
