"""
chroma_client.py

Singleton ChromaDB HTTP client backed directly by ``httpx``.
Connects to a ChromaDB instance running as a standalone service over its
REST API (v2 with automatic v1 fallback), avoiding the ``chromadb-client``
SDK's transitive ``orjson`` / ``grpcio`` / ``opentelemetry`` binary extensions
so Odysseus stays 100% compatible with free-threaded Python (3.13t / 3.14t).
"""

from __future__ import annotations

import logging
import os
import socket
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote, urlparse

import httpx

logger = logging.getLogger(__name__)

_client: Optional["ChromaHttpClient"] = None

DEFAULT_TENANT = "default_tenant"
DEFAULT_DATABASE = "default_database"

# A short connect probe so an unreachable ChromaDB fails fast instead of
# blocking on the OS connection timeout (~30-60s, WinError 10060 on Windows),
# which otherwise stalls app startup. Tunable via CHROMADB_CONNECT_TIMEOUT.
_CONNECT_TIMEOUT = float(os.getenv("CHROMADB_CONNECT_TIMEOUT", "2.0"))
_REQUEST_TIMEOUT = float(os.getenv("CHROMADB_REQUEST_TIMEOUT", "30.0"))


def _port_open(host: str, port: int, timeout: Optional[float] = None) -> bool:
    """Return True if a TCP connection to host:port succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout or _CONNECT_TIMEOUT):
            return True
    except OSError:
        return False


def _to_jsonable(value: Any) -> Any:
    """Convert numpy arrays / sequences into standard JSON-serializable lists."""
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


class ChromaHttpCollection:
    """Lightweight ChromaDB collection handle backed by ``ChromaHttpClient``."""

    def __init__(
        self,
        client: "ChromaHttpClient",
        collection_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._client = client
        self.id = str(collection_id)
        self.name = str(name)
        self.metadata: Dict[str, Any] = dict(metadata) if isinstance(metadata, dict) else {}

    def _item_path(self, action: str) -> str:
        return self._client._collections_path(f"/{quote(self.id, safe='')}/{action}")

    def count(self) -> int:
        params = {"read_level": "index_and_wal"} if self._client.api_version == "v2" else None
        resp = self._client._request("GET", self._item_path("count"), params=params)
        if isinstance(resp, dict) and "count" in resp:
            return int(resp["count"])
        return int(resp or 0)

    def add(
        self,
        ids: Sequence[str],
        embeddings: Any,
        documents: Optional[Sequence[Optional[str]]] = None,
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        uris: Optional[Sequence[Optional[str]]] = None,
    ) -> bool:
        payload = {
            "ids": list(ids),
            "embeddings": _to_jsonable(embeddings),
            "metadatas": list(metadatas) if metadatas is not None else None,
            "documents": list(documents) if documents is not None else None,
            "uris": list(uris) if uris is not None else None,
        }
        self._client._request("POST", self._item_path("add"), json_body=payload)
        return True

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Any,
        documents: Optional[Sequence[Optional[str]]] = None,
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        uris: Optional[Sequence[Optional[str]]] = None,
    ) -> bool:
        payload = {
            "ids": list(ids),
            "embeddings": _to_jsonable(embeddings),
            "metadatas": list(metadatas) if metadatas is not None else None,
            "documents": list(documents) if documents is not None else None,
            "uris": list(uris) if uris is not None else None,
        }
        self._client._request("POST", self._item_path("upsert"), json_body=payload)
        return True

    def update(
        self,
        ids: Sequence[str],
        embeddings: Optional[Any] = None,
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        documents: Optional[Sequence[Optional[str]]] = None,
        uris: Optional[Sequence[Optional[str]]] = None,
    ) -> bool:
        payload = {
            "ids": list(ids),
            "embeddings": _to_jsonable(embeddings) if embeddings is not None else None,
            "metadatas": list(metadatas) if metadatas is not None else None,
            "documents": list(documents) if documents is not None else None,
            "uris": list(uris) if uris is not None else None,
        }
        self._client._request("POST", self._item_path("update"), json_body=payload)
        return True

    def get(
        self,
        ids: Optional[Sequence[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        default_include = ["metadatas", "documents"]
        raw_include = default_include if include is None else list(include)
        filtered_include = [i for i in raw_include if i != "data"]
        payload = {
            "ids": list(ids) if ids is not None else None,
            "where": where,
            "limit": limit,
            "offset": offset,
            "where_document": where_document,
            "include": filtered_include,
        }
        resp = self._client._request("POST", self._item_path("get"), json_body=payload) or {}
        return {
            "ids": resp.get("ids") or [],
            "embeddings": resp.get("embeddings"),
            "metadatas": resp.get("metadatas"),
            "documents": resp.get("documents"),
            "uris": resp.get("uris"),
            "data": None,
            "included": raw_include,
        }

    def query(
        self,
        query_embeddings: Any,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[Sequence[str]] = None,
        ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        default_include = ["metadatas", "documents", "distances"]
        raw_include = default_include if include is None else list(include)
        filtered_include = [i for i in raw_include if i != "data"]
        payload = {
            "ids": list(ids) if ids is not None else None,
            "query_embeddings": _to_jsonable(query_embeddings),
            "n_results": int(n_results),
            "where": where,
            "where_document": where_document,
            "include": filtered_include,
        }
        resp = self._client._request("POST", self._item_path("query"), json_body=payload) or {}
        return {
            "ids": resp.get("ids") or [[]],
            "distances": resp.get("distances"),
            "embeddings": resp.get("embeddings"),
            "metadatas": resp.get("metadatas"),
            "documents": resp.get("documents"),
            "uris": resp.get("uris"),
            "data": None,
            "included": raw_include,
        }

    def delete(
        self,
        ids: Optional[Sequence[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "ids": list(ids) if ids is not None else None,
            "where": where,
            "where_document": where_document,
        }
        if limit is not None:
            payload["limit"] = int(limit)
        resp = self._client._request("POST", self._item_path("delete"), json_body=payload)
        deleted = resp.get("deleted", 0) if isinstance(resp, dict) else 0
        return {"deleted": deleted}


class ChromaHttpClient:
    """Pure-``httpx`` ChromaDB HTTP REST client (v2 with automatic v1 fallback)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8100,
        tenant: str = DEFAULT_TENANT,
        database: str = DEFAULT_DATABASE,
        timeout: float = _REQUEST_TIMEOUT,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        if host.startswith(("http://", "https://")):
            parsed = urlparse(host)
            scheme = parsed.scheme or "http"
            netloc = parsed.netloc or f"{parsed.hostname or 'localhost'}:{parsed.port or port}"
            base_url = f"{scheme}://{netloc}"
        else:
            base_url = f"http://{host}:{port}"

        self.base_url = base_url
        self.tenant = tenant or DEFAULT_TENANT
        self.database = database or DEFAULT_DATABASE
        self.api_version = "v2"
        self._session = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            headers={"Content-Type": "application/json"},
        )

    def _collections_path(self, suffix: str = "") -> str:
        if self.api_version == "v1":
            return f"/api/v1/collections{suffix}"
        tenant_q = quote(self.tenant, safe="")
        db_q = quote(self.database, safe="")
        return f"/api/v2/tenants/{tenant_q}/databases/{db_q}/collections{suffix}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        clean_params = {k: v for k, v in (params or {}).items() if v is not None} or None
        resp = self._session.request(method, path, params=clean_params, json=json_body)
        if resp.status_code >= 400:
            detail = resp.text
            try:
                body = resp.json()
                if isinstance(body, dict) and body.get("message"):
                    detail = f"{body.get('error', 'ChromaError')}: {body['message']}"
            except Exception:
                pass
            raise RuntimeError(f"ChromaDB HTTP {resp.status_code} on {method} {path}: {detail}")
        if not resp.content:
            return None
        return resp.json()

    def heartbeat(self) -> int:
        """Verify server liveness and negotiate API v2 vs v1."""
        resp = self._session.get("/api/v2/heartbeat")
        if resp.status_code in (404, 405, 410):
            resp_v1 = self._session.get("/api/v1/heartbeat")
            resp_v1.raise_for_status()
            self.api_version = "v1"
            body = resp_v1.json() if resp_v1.content else {}
        else:
            resp.raise_for_status()
            self.api_version = "v2"
            body = resp.json() if resp.content else {}
        if isinstance(body, dict):
            return int(body.get("nanosecond heartbeat", 1))
        return int(body or 1)

    def _model_to_collection(
        self,
        payload: Dict[str, Any],
        fallback_name: str = "",
        fallback_metadata: Optional[Dict[str, Any]] = None,
    ) -> ChromaHttpCollection:
        return ChromaHttpCollection(
            client=self,
            collection_id=str(payload["id"]),
            name=str(payload.get("name") or fallback_name),
            metadata=payload.get("metadata") if payload.get("metadata") is not None else fallback_metadata,
        )

    def get_collection(self, name: str) -> ChromaHttpCollection:
        params = {"tenant": self.tenant, "database": self.database} if self.api_version == "v1" else None
        data = self._request("GET", self._collections_path(f"/{quote(name, safe='')}"), params=params)
        return self._model_to_collection(data, fallback_name=name)

    def create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        get_or_create: bool = False,
    ) -> ChromaHttpCollection:
        params = {"tenant": self.tenant, "database": self.database} if self.api_version == "v1" else None
        payload: Dict[str, Any] = {
            "name": name,
            "metadata": metadata or None,
            "get_or_create": bool(get_or_create),
        }
        data = self._request("POST", self._collections_path(), params=params, json_body=payload)
        return self._model_to_collection(data, fallback_name=name, fallback_metadata=metadata)

    def get_or_create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChromaHttpCollection:
        return self.create_collection(name=name, metadata=metadata, get_or_create=True)

    def delete_collection(self, name: str) -> None:
        params = {"tenant": self.tenant, "database": self.database} if self.api_version == "v1" else None
        self._request("DELETE", self._collections_path(f"/{quote(name, safe='')}"), params=params)

    def list_collections(self) -> List[ChromaHttpCollection]:
        params = {"tenant": self.tenant, "database": self.database} if self.api_version == "v1" else None
        rows = self._request("GET", self._collections_path(), params=params) or []
        return [self._model_to_collection(r) for r in rows if isinstance(r, dict) and "id" in r]


def get_chroma_client() -> ChromaHttpClient:
    """Get or create the singleton ChromaDB HTTP client."""
    global _client
    if _client is not None:
        return _client

    host = os.getenv("CHROMADB_HOST", "localhost")
    port = int(os.getenv("CHROMADB_PORT", "8100"))

    probe_host = urlparse(host).hostname or host if host.startswith(("http://", "https://")) else host
    if not _port_open(probe_host, port):
        raise RuntimeError(
            f"ChromaDB is not reachable at {probe_host}:{port}. Start the ChromaDB "
            f"service (e.g. `docker compose up chromadb`) or set CHROMADB_HOST / "
            f"CHROMADB_PORT to point at a running instance."
        )

    client = ChromaHttpClient(host=host, port=port)

    # Health check before caching — if the port is open but the service isn't
    # healthy yet (e.g. still starting), don't poison the singleton with a dead
    # client; leave _client unset so the next call retries.
    client.heartbeat()
    _client = client
    logger.info(f"ChromaDB connected: {host}:{port} ({client.api_version})")
    return _client


def reset_client() -> None:
    """Reset the singleton (e.g. after config change)."""
    global _client
    _client = None
