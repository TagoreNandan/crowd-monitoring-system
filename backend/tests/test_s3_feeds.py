"""Unit and integration tests for S3 video feed resolution, security, and health checks."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from fastapi import HTTPException

from backend.app import storage
from backend.app.utils import parse_s3_source, redact_presigned_url
from backend.app.routers.feeds import _resolve_source
from backend.app.routers.devices import _probe_stream_sync
from crowd_monitoring.streaming.frame_source import RTSPSource, FileVideoSource


def test_parse_s3_source():
    # Test s3:// URI format
    bucket, key = parse_s3_source("s3://ground-control-assets-tagore/failsafe/drone_standby.mp4")
    assert bucket == "ground-control-assets-tagore"
    assert key == "failsafe/drone_standby.mp4"

    # Test object key with default bucket
    bucket, key = parse_s3_source("failsafe/drone_standby.mp4", default_bucket="test-bucket")
    assert bucket == "test-bucket"
    assert key == "failsafe/drone_standby.mp4"

    # Test invalid s3:// URI
    with pytest.raises(ValueError):
        parse_s3_source("s3://")


def test_redact_presigned_url():
    raw_url = "https://ground-control-assets-tagore.s3.eu-north-1.amazonaws.com/failsafe/drone_standby.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=secret123"
    redacted = redact_presigned_url(raw_url)
    assert redacted == "https://ground-control-assets-tagore.s3.eu-north-1.amazonaws.com/failsafe/drone_standby.mp4"
    assert "secret123" not in redacted


def test_rtspsource_redacts_presigned_url_description():
    signed_url = "https://bucket.s3.amazonaws.com/video.mp4?X-Amz-Signature=secret_sig_value"
    source = RTSPSource(url=signed_url, display_name="s3://bucket/video.mp4")
    # Simulate open metadata check
    with patch("cv2.VideoCapture") as mock_cap:
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.get.return_value = 30.0
        mock_cap.return_value = mock_cap_instance
        meta = source.open()
        assert meta.description == "s3://bucket/video.mp4"
        assert "secret_sig_value" not in meta.description


def test_presigned_url_generation_mocked(monkeypatch):
    monkeypatch.setenv("S3_BUCKET_NAME", "ground-control-assets-tagore")
    monkeypatch.setenv("AWS_REGION", "eu-north-1")

    mock_client = MagicMock()
    mock_client.generate_presigned_url.return_value = "https://ground-control-assets-tagore.s3.eu-north-1.amazonaws.com/failsafe/drone_standby.mp4?signed=1"

    with patch.object(storage, "_client", return_value=mock_client):
        url = storage.presigned_url(
            key="failsafe/drone_standby.mp4",
            expires_in=86400,
            response_content_type="video/mp4",
        )
        assert "failsafe/drone_standby.mp4" in url
        mock_client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={
                "Bucket": "ground-control-assets-tagore",
                "Key": "failsafe/drone_standby.mp4",
                "ResponseContentType": "video/mp4",
            },
            ExpiresIn=86400,
        )


def test_resolve_source_s3_uri(monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "s3")
    monkeypatch.setenv("S3_BUCKET_NAME", "ground-control-assets-tagore")

    with patch.object(storage, "is_s3_enabled", return_value=True), patch.object(
        storage, "presigned_url", return_value="https://bucket.s3.amazonaws.com/failsafe/drone_standby.mp4?token=123"
    ):
        source = _resolve_source("s3", "s3://ground-control-assets-tagore/failsafe/drone_standby.mp4")
        assert isinstance(source, RTSPSource)


def test_resolve_source_backward_compatibility():
    # RTSP source
    source = _resolve_source("rtsp", "rtsp://192.168.1.100:554/live")
    assert isinstance(source, RTSPSource)
    assert source._url == "rtsp://192.168.1.100:554/live"

    # Camera source
    source_cam = _resolve_source("camera", "0")
    assert source_cam is not None


def test_probe_stream_sync_s3(monkeypatch):
    with patch.object(storage, "is_s3_enabled", return_value=True), patch.object(
        storage, "presigned_url", return_value="https://bucket.s3.amazonaws.com/failsafe/drone_standby.mp4?token=123"
    ), patch("cv2.VideoCapture") as mock_cap:
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        import numpy as np
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cap_instance.read.return_value = (True, dummy_frame)
        mock_cap_instance.get.return_value = 30.0
        mock_cap.return_value = mock_cap_instance

        res = _probe_stream_sync("failsafe/drone_standby.mp4", "S3")
        assert res["is_reachable"] is True
        assert res["resolution"] == "1920x1080"
        assert res["fps"] == 30.0
        assert "token" not in str(res)
