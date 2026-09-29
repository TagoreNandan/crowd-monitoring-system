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
import time
from pathlib import Path
from typing import Any


def _load_env_if_present() -> None:
    """Read root .env file into os.environ if not already set."""
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


_load_env_if_present()


def is_s3_enabled() -> bool:
    _load_env_if_present()
    return os.environ.get("STORAGE_BACKEND", "local").lower() == "s3"


def _client():
    import importlib

    boto3 = importlib.import_module("boto3")  # local dynamic import: only required when S3 mode is used

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



def get_default_ttl() -> int:
    """Get the default pre-signed URL TTL in seconds from S3_PRESIGNED_TTL env var (default: 300s / 5 mins)."""
    _load_env_if_present()
    try:
        return int(os.environ.get("S3_PRESIGNED_TTL", "300"))
    except ValueError:
        return 300


def presigned_url(key: str, expires_in: int | None = None) -> str:
    """A temporary, signed URL the browser can GET directly from S3 --
    used instead of streaming the file through FastAPI."""
    if expires_in is None:
        expires_in = get_default_ttl()
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": _bucket(), "Key": key},
        ExpiresIn=expires_in,
    )


def get_presigned_url_info(key: str, expires_in: int | None = None) -> dict[str, Any]:
    """Return pre-signed URL along with TTL and expiration timestamp for frontend auto-renewal."""
    if expires_in is None:
        expires_in = get_default_ttl()
    url = presigned_url(key, expires_in=expires_in)
    now = int(time.time())
    return {
        "url": url,
        "expires_in": expires_in,
        "expires_at": now + expires_in,
    }


def delete_object(key: str) -> None:
    _client().delete_object(Bucket=_bucket(), Key=key)
