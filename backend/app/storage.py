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


# Region of the ground-control-assets-tagore bucket.
S3_REGION = "ap-southeast-2"


def is_s3_enabled() -> bool:
    _load_env_if_present()
    try:
        import importlib.util
        if importlib.util.find_spec("boto3") is None:
            return False
        # If STORAGE_BACKEND=s3, or S3_BUCKET_NAME is set, S3 integration is available
        storage_mode = os.environ.get("STORAGE_BACKEND", "local").lower()
        has_bucket = bool(os.environ.get("S3_BUCKET_NAME", "ground-control-assets-tagore"))
        return storage_mode == "s3" or has_bucket
    except Exception:
        return False


def _client():
    import importlib

    boto3 = importlib.import_module("boto3")  # local dynamic import: only required when S3 mode is used
    _load_env_if_present()

    profile = os.environ.get("AWS_PROFILE")

    # Pinned rather than read from AWS_REGION / ~/.aws/config: a mismatched
    # machine default makes S3 answer PermanentRedirect for this bucket.
    session_kwargs: dict[str, Any] = {"region_name": S3_REGION}
    if profile:
        session_kwargs["profile_name"] = profile

    session = boto3.Session(**session_kwargs)

    client_kwargs: dict[str, Any] = {}
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        client_kwargs["aws_access_key_id"] = os.environ["AWS_ACCESS_KEY_ID"]
    if os.environ.get("AWS_SECRET_ACCESS_KEY"):
        client_kwargs["aws_secret_access_key"] = os.environ["AWS_SECRET_ACCESS_KEY"]
    if os.environ.get("AWS_SESSION_TOKEN"):
        client_kwargs["aws_session_token"] = os.environ["AWS_SESSION_TOKEN"]

    client_kwargs["endpoint_url"] = f"https://s3.{S3_REGION}.amazonaws.com"

    return session.client("s3", region_name=S3_REGION, **client_kwargs)


def _bucket() -> str:
    _load_env_if_present()
    return os.environ.get("S3_BUCKET_NAME", "ground-control-assets-tagore")


def upload_file(local_path: Path, key: str, bucket: str | None = None) -> None:
    """Upload a finished local file to S3 under `key`, unchanged bytes."""
    target_bucket = bucket or _bucket()
    _client().upload_file(str(local_path), target_bucket, key)


def upload_bytes(data: bytes, key: str, content_type: str = "application/octet-stream", bucket: str | None = None) -> None:
    """Upload in-memory bytes to S3 as-is."""
    target_bucket = bucket or _bucket()
    _client().put_object(Bucket=target_bucket, Key=key, Body=data, ContentType=content_type)


def object_exists(key: str, bucket: str | None = None) -> bool:
    """Return whether an S3 object exists, propagating non-not-found errors."""
    target_bucket = bucket or _bucket()
    try:
        _client().head_object(Bucket=target_bucket, Key=key)
        return True
    except Exception as exc:
        response = getattr(exc, "response", {})
        error = response.get("Error", {}) if isinstance(response, dict) else {}
        code = str(error.get("Code", ""))
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode") if isinstance(response, dict) else None
        if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
            return False
        raise


def get_default_ttl() -> int:
    """Get the default pre-signed URL TTL in seconds from S3_PRESIGNED_TTL env var (default: 300s / 5 mins)."""
    _load_env_if_present()
    try:
        return int(os.environ.get("S3_PRESIGNED_TTL", "300"))
    except ValueError:
        return 300


def presigned_url(
    key: str,
    expires_in: int | None = None,
    response_disposition: str | None = None,
    response_content_type: str | None = None,
    bucket: str | None = None,
) -> str:
    """A temporary, signed URL the browser or OpenCV can GET directly from S3."""
    if expires_in is None:
        expires_in = get_default_ttl()
    target_bucket = bucket or _bucket()
    params: dict[str, Any] = {"Bucket": target_bucket, "Key": key}
    if response_disposition:
        params["ResponseContentDisposition"] = f'attachment; filename="{response_disposition}"'
    if response_content_type:
        params["ResponseContentType"] = response_content_type
    return _client().generate_presigned_url(
        "get_object",
        Params=params,
        ExpiresIn=expires_in,
    )


def get_presigned_url_info(
    key: str,
    expires_in: int | None = None,
    response_disposition: str | None = None,
    response_content_type: str | None = None,
    bucket: str | None = None,
) -> dict[str, Any]:
    """Return pre-signed URL along with TTL and expiration timestamp for frontend auto-renewal."""
    if expires_in is None:
        expires_in = get_default_ttl()
    url = presigned_url(
        key,
        expires_in=expires_in,
        response_disposition=response_disposition,
        response_content_type=response_content_type,
        bucket=bucket,
    )
    now = int(time.time())
    return {
        "url": url,
        "expires_in": expires_in,
        "expires_at": now + expires_in,
    }


def delete_object(key: str, bucket: str | None = None) -> None:
    target_bucket = bucket or _bucket()
    _client().delete_object(Bucket=target_bucket, Key=key)

