"""Sanity check: every module in the package imports cleanly."""

import importlib


def test_import_crowd_monitoring():
    import crowd_monitoring  # noqa: F401


def test_import_submodules():
    for mod in [
        "crowd_monitoring.device",
        "crowd_monitoring.models",
        "crowd_monitoring.models.dmcount",
        "crowd_monitoring.models.csrnet",
        "crowd_monitoring.models.weights",
        "crowd_monitoring.inference",
        "crowd_monitoring.heatmap",
        "crowd_monitoring.localization",
        "crowd_monitoring.clustering",
        "crowd_monitoring.risk",
        "crowd_monitoring.evaluation",
        "crowd_monitoring.video",
        "crowd_monitoring.system_status",
        "crowd_monitoring.pipeline",
    ]:
        importlib.import_module(mod)
