"""Load a YAML config file into the pieces the pipeline needs.

Every value has a hardcoded default already (see risk.RiskConfig and
pipeline.CrowdAnalysisPipeline's constructor arguments); this module is
just an optional convenience for overriding them from one file instead of
a pile of CLI flags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .risk import RiskConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def risk_config_from_dict(cfg: dict) -> RiskConfig:
    risk_cfg = cfg.get("risk", {})
    return RiskConfig(**{k: v for k, v in risk_cfg.items() if hasattr(RiskConfig, k)})


def pipeline_kwargs_from_config(cfg: dict) -> dict:
    """Build the kwargs CrowdAnalysisPipeline expects from a parsed config dict."""
    model_cfg = cfg.get("model", {})
    cluster_cfg = cfg.get("clustering", {})
    return {
        "model_name": model_cfg.get("name", "DM-Count"),
        "model_weights": model_cfg.get("weights", "QNRF"),
        "device": model_cfg.get("device", "auto"),
        "max_side": model_cfg.get("max_side", 1024),
        "eps_px": cluster_cfg.get("eps_px", 40),
        "min_samples": cluster_cfg.get("min_samples", 3),
        "risk_config": risk_config_from_dict(cfg),
    }
