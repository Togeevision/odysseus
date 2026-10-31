#!/usr/bin/env python3
"""Project private ArchiveBox snapshots into a local Typesense collection."""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, urlparse


COLLECTION_NAME = "cultural_archive"
TEXT_CANDIDATES = (
    "readability/content.txt",
    "mercury/content.txt",
    "htmltotext.txt",
)
FACET_TAGS = {
    "city": "cityKey",
    "entity-type": "entityType",
    "entity": "entityId",
    "watch": "watchId",
    "fingerprint": "eventFingerprint",
}
COLLECTION_SCHEMA = {
    "name": COLLECTION_NAME,
    "fields": [
        {"name": "id", "type": "string"},
        {"name": "timestamp", "type": "string"},
        {"name": "archived_at", "type": "int64", "sort": True},
        {"name": "url", "type": "string"},
        {"name": "domain", "type": "string", "facet": True},
        {"name": "title", "type": "string"},
        {"name": "content", "type": "string"},
        {"name": "contentHash", "type": "string"},
        {"name": "tags", "type": "string[]", "facet": True, "optional": True},
        {"name": "cityKey", "type": "string", "facet": True, "optional": True},
        {"name": "entityType", "type": "string", "facet": True, "optional": True},
        {"name": "entityId", "type": "string", "facet": True, "optional": True},
        {"name": "watchId", "type": "string", "facet": True, "optional": True},
        {"name": "eventFingerprint", "type": "string", "optional": True},
        {"name": "archivePath", "type": "string"},
    ],
    "default_sorting_field": "archived_at",
}


def _bounded_positive_int(value: str | int, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, 1), maximum)


def parse_tags(meta: dict[str, Any]) -> list[str]:
    raw = meta.get("tags")
    if isinstance(raw, list):
        tags = [str(item.get("name", "") if isinstance(item, dict) else item) for item in raw]
    else:
        raw = meta.get("tags_str") or raw or ""
        tags = re.split(r"[,\n]", str(raw))
    return sorted({tag.strip()[:160] for tag in tags if tag and tag.strip()})


def tag_facets(tags: Iterable[str]) -> dict[str, str]:
    facets: dict[str, str] = {}
    for tag in tags:
        prefix, separator, value = tag.partition(":")
        field = FACET_TAGS.get(prefix.strip().lower())
        if separator and field and value.strip() and field not in facets:
            facets[field] = value.strip()[:160]
    return facets


def extract_best_text(snapshot_dir: Path, max_chars: int) -> str:
    for relative_path in TEXT_CANDIDATES:
        candidate = snapshot_dir / relative_path
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                return text[:max_chars]
    return ""


def archived_at(timestamp: str, meta: dict[str, Any]) -> int:
    for raw in (meta.get("bookmarked_at"), meta.get("added"), meta.get("created_at")):
        if isinstance(raw, str):
            try:
                return int(calendar.timegm(time.strptime(raw[:19], "%Y-%m-%dT%H:%M:%S")))
            except ValueError:
                pass
    digits = re.sub(r"\D", "", timestamp)
    if len(digits) >= 14:
        try:
            return int(calendar.timegm(time.strptime(digits[:14], "%Y%m%d%H%M%S")))
        except ValueError:
            pass
    return int(timestamp) if timestamp.isdigit() and len(timestamp) <= 10 else 0


def build_document(index_file: Path, archive_root: Path, max_chars: int) -> dict[str, Any]:
    meta = json.loads(index_file.read_text(encoding="utf-8"))
    snapshot_dir = index_file.parent
    url = str(meta.get("url") or meta.get("original_url") or "").strip()
    timestamp = str(meta.get("timestamp") or snapshot_dir.name).strip()
    if not url or not timestamp:
        raise ValueError("snapshot index is missing url or timestamp")
    content = extract_best_text(snapshot_dir, max_chars)
    tags = parse_tags(meta)
    stable_id = hashlib.sha256(f"{timestamp}\0{url}".encode("utf-8")).hexdigest()
    relative_path = snapshot_dir.relative_to(archive_root.parent).as_posix()
    document: dict[str, Any] = {
        "id": stable_id,
        "timestamp": timestamp[:64],
        "archived_at": archived_at(timestamp, meta),
        "url": url[:4096],
        "domain": (urlparse(url).hostname or "")[:255].lower(),
        "title": str(meta.get("title") or url)[:500],
        "content": content,
        "contentHash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "tags": tags,
        "archivePath": f"/{relative_path}/index.html",
    }
    document.update(tag_facets(tags))
    return document


def iter_index_files(data_dir: Path) -> Iterable[Path]:
    archive_root = data_dir / "archive"
    if not archive_root.is_dir():
        return []
    return sorted(archive_root.rglob("index.json"))


def request_json(base_url: str, api_key: str, path: str, *, method: str = "GET", body: bytes | None = None, content_type: str = "application/json") -> tuple[int, str]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        method=method,
        headers={"X-TYPESENSE-API-KEY": api_key, "Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")


def ensure_collection(base_url: str, api_key: str) -> None:
    status, _ = request_json(base_url, api_key, f"/collections/{quote(COLLECTION_NAME)}")
    if status == 200:
        return
    if status != 404:
        raise RuntimeError(f"Typesense collection lookup failed with HTTP {status}")
    status, response = request_json(
        base_url,
        api_key,
        "/collections",
        method="POST",
        body=json.dumps(COLLECTION_SCHEMA).encode("utf-8"),
    )
    if status not in (200, 201, 409):
        raise RuntimeError(f"Typesense collection creation failed with HTTP {status}: {response[:300]}")


def import_documents(base_url: str, api_key: str, documents: list[dict[str, Any]]) -> tuple[int, int]:
    if not documents:
        return 0, 0
    payload = "\n".join(json.dumps(document, separators=(",", ":")) for document in documents).encode("utf-8")
    status, response = request_json(
        base_url,
        api_key,
        f"/collections/{quote(COLLECTION_NAME)}/documents/import?action=upsert",
        method="POST",
        body=payload,
        content_type="text/plain",
    )
    if status != 200:
        raise RuntimeError(f"Typesense import failed with HTTP {status}: {response[:300]}")
    results = [json.loads(line) for line in response.splitlines() if line.strip()]
    succeeded = sum(1 for result in results if result.get("success") is True)
    return succeeded, len(results) - succeeded


def run_once(data_dir: Path, base_url: str, api_key: str, max_chars: int) -> int:
    ensure_collection(base_url, api_key)
    archive_root = data_dir / "archive"
    documents: list[dict[str, Any]] = []
    skipped = 0
    for index_file in iter_index_files(data_dir):
        try:
            documents.append(build_document(index_file, archive_root, max_chars))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            skipped += 1
            print(f"[archive-indexer] skipped {index_file}: {error}", file=sys.stderr)
    succeeded, failed = import_documents(base_url, api_key, documents)
    print(f"[archive-indexer] indexed={succeeded} failed={failed} skipped={skipped}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="repeat indexing at the configured interval")
    args = parser.parse_args()
    data_dir = Path(os.environ.get("ARCHIVEBOX_DATA_DIR", "/archivebox"))
    base_url = os.environ.get("TYPESENSE_URL", "http://127.0.0.1:8108")
    api_key = os.environ.get("TYPESENSE_API_KEY", "")
    if not api_key:
        parser.error("TYPESENSE_API_KEY is required")
    max_chars = _bounded_positive_int(os.environ.get("ARCHIVE_INDEX_MAX_CONTENT_CHARS", "100000"), 100000, 500000)
    interval = _bounded_positive_int(os.environ.get("ARCHIVE_INDEX_INTERVAL_SECONDS", "300"), 300, 86400)
    while True:
        result = run_once(data_dir, base_url, api_key, max_chars)
        if not args.watch:
            return result
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
