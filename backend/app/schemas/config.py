"""Response schema for GET /api/config.

Mirrors `crowdmon.risk.RiskConfig` field-for-field on purpose: this is the
real, live threshold set the risk engine is using, not a re-derived summary.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ModelConfigSchema(BaseModel):
    name: Optional[str]
    weights: Optional[str]
    device: str
    device_resolved: bool
    max_side: Optional[int]


class ClusteringConfigSchema(BaseModel):
    eps_px: Optional[float]
    min_samples: Optional[int]


class RiskConfigSchema(BaseModel):
    count_caution: float
    count_warning: float
    count_critical: float
    density_caution: float
    density_warning: float
    density_critical: float
    concentration_caution: float
    concentration_warning: float
    concentration_critical: float
    growth_caution: float
    growth_warning: float
    growth_critical: float
    weight_count: float
    weight_density: float
    weight_concentration: float
    weight_growth: float


class AppConfigResponse(BaseModel):
    model: ModelConfigSchema
    clustering: ClusteringConfigSchema
    risk: RiskConfigSchema
