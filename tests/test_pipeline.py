from crowdmon.pipeline import FrameResult


def test_pipeline_end_to_end(pipeline, sample_image):
    result = pipeline.analyze(sample_image)
    assert isinstance(result, FrameResult)
    assert result.inference.count >= 0
    assert result.heatmap_bgr.shape[:2] == sample_image.shape[:2]
    assert result.overlay_bgr.shape == sample_image.shape
    assert result.risk.status in ("SAFE", "CAUTION", "WARNING", "CRITICAL")


def test_pipeline_updates_system_status(pipeline, sample_image):
    from crowdmon.system_status import GLOBAL_STATUS

    pipeline.analyze(sample_image)
    snap = GLOBAL_STATUS.snapshot()
    assert snap.model_loaded is True
    assert snap.frames_processed >= 1
    assert snap.state() in ("ONLINE", "DEGRADED")
