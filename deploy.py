"""
Deployment helpers for turbovec-mcp.

Supported targets:
  - local copy   (default, no extra deps)
  - archive      .tar.gz bundle  (no extra deps)
  - s3://        Amazon S3       (pip install boto3)
  - gs://        Google Cloud Storage  (pip install google-cloud-storage)
  - azure://     Azure Blob Storage    (pip install azure-storage-blob)

Import sources:
  - local .tar.gz path
  - s3:// / gs:// / azure:// URI pointing to a .tar.gz object
"""

from __future__ import annotations

import os
import tarfile
import shutil
from pathlib import Path
from typing import Optional

BASE_DIR = Path.home() / ".turbovec"


# ------------------------------------------------------------------ #
# Archive  (.tar.gz)                                                   #
# ------------------------------------------------------------------ #

def export_archive(db_name: str, output_path: Optional[str] = None) -> str:
    """
    Bundle <db_name>.tvim + <db_name>.json into a single .tar.gz file.
    Returns the absolute path of the created archive.
    """
    src_base = BASE_DIR / db_name
    tvim = src_base.with_suffix(".tvim")
    meta = src_base.with_suffix(".json")

    if not tvim.exists():
        raise FileNotFoundError(f"Database '{db_name}' not found in {BASE_DIR}")

    out = Path(output_path) if output_path else BASE_DIR / f"{db_name}.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(out, "w:gz") as tar:
        tar.add(tvim, arcname=f"{db_name}.tvim")
        if meta.exists():
            tar.add(meta, arcname=f"{db_name}.json")

    return str(out)


def import_archive(archive_path: str, db_name: Optional[str] = None) -> str:
    """
    Extract a .tar.gz archive into ~/.turbovec/.
    db_name overrides the name encoded in the archive.
    Returns the final database name.
    """
    src = Path(archive_path)
    if not src.exists():
        raise FileNotFoundError(archive_path)

    with tarfile.open(src, "r:gz") as tar:
        members = tar.getmembers()
        stems = {Path(m.name).stem for m in members}
        if len(stems) != 1:
            raise ValueError(
                f"Archive contains files from multiple databases: {stems}. "
                "Pass db_name to pick a target name."
            )
        detected = stems.pop()
        target = db_name or detected

        BASE_DIR.mkdir(parents=True, exist_ok=True)
        for member in members:
            ext = Path(member.name).suffix
            dest = BASE_DIR / (target + ext)
            f = tar.extractfile(member)
            if f:
                dest.write_bytes(f.read())

    return target


# ------------------------------------------------------------------ #
# URI parsing                                                          #
# ------------------------------------------------------------------ #

def _parse_uri(uri: str) -> tuple[str, str, str]:
    """
    Parse  s3://bucket/prefix/name
           gs://bucket/prefix/name
        azure://container/prefix/name

    Returns (provider, bucket_or_container, key_prefix_without_extension).
    """
    for scheme, provider in [("s3://", "s3"), ("gs://", "gcs"), ("azure://", "azure")]:
        if uri.startswith(scheme):
            rest = uri[len(scheme):]
            bucket, _, key = rest.partition("/")
            return provider, bucket, key
    raise ValueError(
        f"Unsupported URI '{uri}'. Use s3://, gs://, or azure://"
    )


def _key_base(prefix: str, db_name: str) -> str:
    """Build the base key (no extension): prefix/db_name or db_name."""
    if prefix:
        return prefix.rstrip("/") + "/" + db_name
    return db_name


# ------------------------------------------------------------------ #
# Cloud upload                                                         #
# ------------------------------------------------------------------ #

def upload_cloud(db_name: str, uri: str) -> list[str]:
    """
    Upload <db_name>.tvim + .json to cloud storage.
    URI format: s3://bucket/optional/prefix  (files land at prefix/db_name.{tvim,json})
    Returns list of remote URIs that were written.
    """
    provider, bucket, prefix = _parse_uri(uri)
    src_base = BASE_DIR / db_name

    files: list[tuple[Path, str]] = []
    for ext in (".tvim", ".json"):
        f = src_base.with_suffix(ext)
        if not f.exists():
            raise FileNotFoundError(f"Database '{db_name}' not found (missing {f.name})")
        files.append((f, ext))

    base = _key_base(prefix, db_name)
    uploaded: list[str] = []

    if provider == "s3":
        try:
            import boto3
        except ImportError:
            raise ImportError("S3 upload requires: pip install boto3")
        s3 = boto3.client("s3")
        for src, ext in files:
            key = base + ext
            s3.upload_file(str(src), bucket, key)
            uploaded.append(f"s3://{bucket}/{key}")

    elif provider == "gcs":
        try:
            from google.cloud import storage as gcs
        except ImportError:
            raise ImportError("GCS upload requires: pip install google-cloud-storage")
        client = gcs.Client()
        bkt = client.bucket(bucket)
        for src, ext in files:
            key = base + ext
            bkt.blob(key).upload_from_filename(str(src))
            uploaded.append(f"gs://{bucket}/{key}")

    elif provider == "azure":
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError:
            raise ImportError("Azure upload requires: pip install azure-storage-blob")
        conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if not conn:
            raise ValueError("Set AZURE_STORAGE_CONNECTION_STRING env var for Azure upload")
        svc = BlobServiceClient.from_connection_string(conn)
        cc = svc.get_container_client(bucket)
        for src, ext in files:
            key = base + ext
            with open(src, "rb") as fh:
                cc.upload_blob(key, fh, overwrite=True)
            uploaded.append(f"azure://{bucket}/{key}")

    return uploaded


# ------------------------------------------------------------------ #
# Cloud download                                                       #
# ------------------------------------------------------------------ #

def download_cloud(uri: str, db_name: Optional[str] = None) -> str:
    """
    Download a database from cloud storage into ~/.turbovec/.
    URI must point to the base path (no extension):
        s3://bucket/prefix/mydb  →  downloads mydb.tvim + mydb.json
    Returns the local database name.
    """
    provider, bucket, key_prefix = _parse_uri(uri)

    # derive db name from the last segment of the key prefix
    remote_name = Path(key_prefix).name if key_prefix else None
    target = db_name or remote_name
    if not target:
        raise ValueError(
            "Cannot determine db name from URI. "
            "Either end the URI with the db name or pass db_name explicitly."
        )

    base = _key_base(key_prefix, "") if key_prefix.endswith("/") else key_prefix
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    if provider == "s3":
        try:
            import boto3
        except ImportError:
            raise ImportError("S3 download requires: pip install boto3")
        s3 = boto3.client("s3")
        for ext in (".tvim", ".json"):
            s3.download_file(bucket, base + ext, str(BASE_DIR / (target + ext)))

    elif provider == "gcs":
        try:
            from google.cloud import storage as gcs
        except ImportError:
            raise ImportError("GCS download requires: pip install google-cloud-storage")
        client = gcs.Client()
        bkt = client.bucket(bucket)
        for ext in (".tvim", ".json"):
            bkt.blob(base + ext).download_to_filename(str(BASE_DIR / (target + ext)))

    elif provider == "azure":
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError:
            raise ImportError("Azure download requires: pip install azure-storage-blob")
        conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if not conn:
            raise ValueError("Set AZURE_STORAGE_CONNECTION_STRING env var for Azure download")
        svc = BlobServiceClient.from_connection_string(conn)
        cc = svc.get_container_client(bucket)
        for ext in (".tvim", ".json"):
            dest = BASE_DIR / (target + ext)
            dest.write_bytes(cc.get_blob_client(base + ext).download_blob().readall())

    return target


# ------------------------------------------------------------------ #
# Local copy (original behaviour)                                      #
# ------------------------------------------------------------------ #

def _download_single_cloud_file(provider: str, bucket: str, key: str, dest: str):
    """Download one object from cloud storage to a local path."""
    if provider == "s3":
        import boto3
        boto3.client("s3").download_file(bucket, key, dest)
    elif provider == "gcs":
        from google.cloud import storage as gcs
        gcs.Client().bucket(bucket).blob(key).download_to_filename(dest)
    elif provider == "azure":
        from azure.storage.blob import BlobServiceClient
        conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if not conn:
            raise ValueError("Set AZURE_STORAGE_CONNECTION_STRING for Azure download")
        svc = BlobServiceClient.from_connection_string(conn)
        Path(dest).write_bytes(
            svc.get_container_client(bucket).get_blob_client(key).download_blob().readall()
        )


def copy_local(db_name: str, dest_name: str, output_dir: Optional[str] = None) -> list[str]:
    """Copy a database to a new name, optionally in a different directory."""
    src_base = BASE_DIR / db_name
    dst_dir = Path(output_dir) if output_dir else BASE_DIR
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for ext in (".tvim", ".json"):
        src = src_base.with_suffix(ext)
        if src.exists():
            dst = dst_dir / (dest_name + ext)
            shutil.copy2(src, dst)
            copied.append(str(dst))

    if not copied:
        raise FileNotFoundError(f"Database '{db_name}' not found in {BASE_DIR}")

    return copied
