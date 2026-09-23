"""Optional S3-backed storage for finished job output files.

By default (STORAGE_BACKEND unset or "local"), this module does nothing --
your app behaves exactly as it does today. Setting STORAGE_BACKEND=s3 (plus
the AWS_* / S3_BUCKET_NAME env vars below) turns on uploading finished
video/image output to S3.

boto3 is imported lazily, so this module -- and the app -- import fine even
if boto3 isn't installed, as long as STORAGE_BACKEND stays "local".
"""

from __future__ import annotations

import os
from pathlib import Path


def is_s3_enabled() -> bool:
    return os.environ.get("STORAGE_BACKEND", "local").lower() == "s3"


def _client():
    import boto3  # local import: only required when S3 mode is actually used

    return boto3.client(
        "s3",
        region_name=os.environ["AWS_REGION"],
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def _bucket() -> str:
    return os.environ["S3_BUCKET_NAME"]


def upload_file(local_path: Path, key: str) -> None:
    """Upload a finished local file to S3 under `key`, unchanged bytes.
    Only call this when `is_s3_enabled()` is True."""
    _client().upload_file(str(local_path), _bucket(), key)


def upload_bytes(data: bytes, key: str, content_type: str = "application/octet-stream") -> None:
    """Upload in-memory bytes (e.g. an already-encoded image) to S3 as-is,
    no re-encoding."""
    _client().put_object(Bucket=_bucket(), Key=key, Body=data, ContentType=content_type)


def presigned_url(key: str, expires_in: int = 3600) -> str:
    """A temporary, signed URL the browser can GET directly from S3 --
    used instead of streaming the file through FastAPI."""
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": _bucket(), "Key": key},
        ExpiresIn=expires_in,
    )


def delete_object(key: str) -> None:
    _client().delete_object(Bucket=_bucket(), Key=key)
