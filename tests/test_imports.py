"""Sanity check: every module in the package imports cleanly."""

import importlib


def test_import_crowdmon():
    import crowdmon  # noqa: F401


def test_import_submodules():
    for mod in [
        "crowdmon.device",
        "crowdmon.models",
        "crowdmon.models.dmcount",
        "crowdmon.models.csrnet",
        "crowdmon.models.weights",
        "crowdmon.inference",
        "crowdmon.heatmap",
        "crowdmon.localization",
        "crowdmon.clustering",
        "crowdmon.risk",
        "crowdmon.evaluation",
        "crowdmon.video",
        "crowdmon.system_status",
        "crowdmon.pipeline",
    ]:
        importlib.import_module(mod)
