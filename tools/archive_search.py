#!/usr/bin/env python3
"""Read-only Typesense query helper for Odysseus cultural_archive searches.

Uses a Typesense search-only parent key (actions=['documents:search'] on
collection 'cultural_archive') and derives per-query HMAC-SHA256 scoped search
keys so research tasks can cross-reference preserved snapshots by cityKey or
entityId without access to the bootstrap admin key.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


COLLECTION_NAME = "cultural_archive"
QUERY_BY_FIELDS = "title,content,url,tags"
MAX_SNIPPET_CHARS = 600
SAFE_FILTER_RE = re.compile(r"^[a-zA-Z0-9._:-]{1,160}$")


def _sanitize_filter_token(value: str | None) -> str:
    if not value:
        return ""
    cleaned = str(value).strip()
    if not SAFE_FILTER_RE.match(cleaned):
        raise ValueError(f"Invalid filter token: {cleaned[:40]!r}")
    return cleaned


def build_filter_by(
    *,
    city_key: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    watch_id: str | None = None,
    domain: str | None = None,
) -> str:
    clauses: list[str] = []
    field_map = (
        ("cityKey", city_key),
        ("entityType", entity_type),
        ("entityId", entity_id),
        ("watchId", watch_id),
        ("domain", domain),
    )
    for field_name, raw_val in field_map:
        token = _sanitize_filter_token(raw_val)
        if token:
            clauses.append(f"{field_name}:=`{token}`")
    return " && ".join(clauses)


def generate_scoped_search_key(search_only_key: str, embedded_params: dict[str, Any]) -> str:
    """Derive a Typesense scoped API key via HMAC-SHA256."""
    clean_key = (search_only_key or "").strip()
    if len(clean_key) < 8:
        raise ValueError("search_only_key must be at least 8 characters")
    params_json = json.dumps(embedded_params, separators=(",", ":"), sort_keys=True)
    digest_b64 = base64.b64encode(
        hmac.new(
            clean_key.encode("utf-8"),
            params_json.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    ).decode("utf-8")
    raw_scoped = f"{digest_b64}{clean_key[:4]}{params_json}"
    return base64.b64encode(raw_scoped.encode("utf-8")).decode("utf-8")


def _request_json(
    base_url: str,
    api_key: str,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
) -> tuple[int, str]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        method=method,
        headers={
            "X-TYPESENSE-API-KEY": api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")


def mint_search_only_key(base_url: str, admin_api_key: str) -> str:
    """Create a Typesense search-only key restricted to documents:search on cultural_archive."""
    clean_admin = (admin_api_key or "").strip()
    if not clean_admin:
        raise ValueError("TYPESENSE_API_KEY is required to mint a search-only key")
    payload = {
        "description": "Odysseus/OpenClaw read-only search key for cultural_archive",
        "actions": ["documents:search"],
        "collections": [COLLECTION_NAME],
    }
    status, body = _request_json(
        base_url,
        clean_admin,
        "/keys",
        method="POST",
        body=json.dumps(payload).encode("utf-8"),
    )
    if status not in (200, 201):
        raise RuntimeError(f"Typesense key creation failed with HTTP {status}: {body[:200]}")
    parsed = json.loads(body)
    key_value = str(parsed.get("value") or "").strip()
    if not key_value:
        raise RuntimeError("Typesense did not return a key value")
    return key_value


def search_cultural_archive(
    *,
    query: str = "*",
    city_key: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    watch_id: str | None = None,
    domain: str | None = None,
    limit: int = 10,
    base_url: str | None = None,
    search_only_key: str | None = None,
) -> dict[str, Any]:
    """Execute a read-only search against cultural_archive using a scoped search key."""
    resolved_url = (base_url or os.environ.get("TYPESENSE_URL") or "http://127.0.0.1:8108").strip()
    resolved_key = (search_only_key or os.environ.get("TYPESENSE_SEARCH_ONLY_KEY") or "").strip()
    bootstrap_key = (os.environ.get("TYPESENSE_API_KEY") or "").strip()

    if not resolved_key:
        raise ValueError("TYPESENSE_SEARCH_ONLY_KEY is required for read-only archive queries")
    if bootstrap_key and resolved_key == bootstrap_key:
        raise ValueError("TYPESENSE_SEARCH_ONLY_KEY must be a search-only key, not the bootstrap TYPESENSE_API_KEY")

    bounded_limit = min(max(int(limit or 10), 1), 25)
    filter_by = build_filter_by(
        city_key=city_key,
        entity_type=entity_type,
        entity_id=entity_id,
        watch_id=watch_id,
        domain=domain,
    )

    embedded_params: dict[str, Any] = {
        "query_by": QUERY_BY_FIELDS,
        "per_page": bounded_limit,
    }
    if filter_by:
        embedded_params["filter_by"] = filter_by

    scoped_key = generate_scoped_search_key(resolved_key, embedded_params)

    query_params: dict[str, str] = {
        "q": (query or "*").strip() or "*",
        "query_by": QUERY_BY_FIELDS,
        "sort_by": "archived_at:desc",
        "per_page": str(bounded_limit),
        "facet_by": "cityKey,entityType,entityId,domain",
    }
    if filter_by:
        query_params["filter_by"] = filter_by

    encoded_query = urllib.parse.urlencode(query_params)
    status, response_text = _request_json(
        resolved_url,
        scoped_key,
        f"/collections/{urllib.parse.quote(COLLECTION_NAME)}/documents/search?{encoded_query}",
    )
    if status == 404:
        return {"found": 0, "hits": [], "facets": {}}
    if status != 200:
        raise RuntimeError(f"Typesense search failed with HTTP {status}: {response_text[:200]}")

    payload = json.loads(response_text)
    hits: list[dict[str, Any]] = []
    for raw_hit in payload.get("hits") or []:
        doc = raw_hit.get("document") or {}
        content = str(doc.get("content") or "")
        hits.append(
            {
                "id": doc.get("id"),
                "title": doc.get("title"),
                "url": doc.get("url"),
                "domain": doc.get("domain"),
                "timestamp": doc.get("timestamp"),
                "archived_at": doc.get("archived_at"),
                "cityKey": doc.get("cityKey"),
                "entityType": doc.get("entityType"),
                "entityId": doc.get("entityId"),
                "watchId": doc.get("watchId"),
                "contentHash": doc.get("contentHash"),
                "archivePath": doc.get("archivePath"),
                "snippet": content[:MAX_SNIPPET_CHARS],
            }
        )

    facets: dict[str, list[dict[str, Any]]] = {}
    for facet in payload.get("facet_stats") or payload.get("facet_counts") or []:
        field_name = facet.get("field_name")
        if field_name:
            facets[field_name] = [
                {"value": item.get("value"), "count": item.get("count")}
                for item in (facet.get("counts") or [])
            ]

    return {
        "found": int(payload.get("found") or len(hits)),
        "hits": hits,
        "facets": facets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="*", help="search query string (default: *)")
    parser.add_argument("--city", dest="city_key", help="filter by cityKey facet")
    parser.add_argument("--entity-type", dest="entity_type", help="filter by entityType facet")
    parser.add_argument("--entity", dest="entity_id", help="filter by entityId facet")
    parser.add_argument("--watch", dest="watch_id", help="filter by watchId facet")
    parser.add_argument("--domain", help="filter by domain facet")
    parser.add_argument("--limit", type=int, default=10, help="max results (1-25)")
    parser.add_argument(
        "--mint-search-key",
        action="store_true",
        help="mint a new documents:search key on cultural_archive using TYPESENSE_API_KEY",
    )
    args = parser.parse_args()

    base_url = os.environ.get("TYPESENSE_URL", "http://127.0.0.1:8108")
    if args.mint_search_key:
        admin_key = os.environ.get("TYPESENSE_API_KEY", "")
        key = mint_search_only_key(base_url, admin_key)
        print(key)
        return 0

    result = search_cultural_archive(
        query=args.query,
        city_key=args.city_key,
        entity_type=args.entity_type,
        entity_id=args.entity_id,
        watch_id=args.watch_id,
        domain=args.domain,
        limit=args.limit,
        base_url=base_url,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
