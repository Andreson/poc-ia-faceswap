"""Thin boto3 wrapper for Cloudflare R2 (S3-compatible). See REQ-ARCH-006, REQ-AI-012.

Used by handler.py to download job inputs from R2 before processing and
upload the final output video back to R2.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import boto3

TMP_DIR = Path("/tmp")


def _client():
    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _bucket() -> str:
    return os.environ["R2_BUCKET_NAME"]


def download_from_r2(key: str) -> Path:
    local_path = TMP_DIR / f"{uuid4().hex}_{Path(key).name}"
    _client().download_file(_bucket(), key, str(local_path))
    return local_path


def upload_to_r2(local_path: Path, key: str) -> str:
    _client().upload_file(str(local_path), _bucket(), key)
    return key
