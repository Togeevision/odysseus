"""Regression and protocol tests for the pure-httpx ChromaDB client (issue #326).

Covers:
1. Fast-fail TCP preflight so an unreachable ChromaDB doesn't block startup.
2. Failed connection / heartbeat must not poison the cached singleton.
3. Chroma v2 REST API + automatic v1 fallback for collection CRUD, count,
   add, upsert, update, get, query, delete, and numpy ndarray serialization.
"""
import json
import socket
import time

import httpx
import numpy as np
import pytest

import src.chroma_client as cc


def _free_port() -> int:
    """Bind to port 0, grab the assigned port, release it — nothing listens."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_port_open_false_for_closed_port_and_is_fast():
    port = _free_port()
    t0 = time.monotonic()
    assert cc._port_open("127.0.0.1", port, timeout=1.0) is False
    # The whole point: we fail fast, nowhere near the 30-60s OS timeout.
    assert time.monotonic() - t0 < 5.0


def test_port_open_true_for_listening_socket():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()
    try:
        assert cc._port_open(host, port, timeout=1.0) is True
    finally:
        srv.close()


def test_get_chroma_client_does_not_cache_when_unreachable(monkeypatch):
    cc.reset_client()
    monkeypatch.setenv("CHROMADB_HOST", "127.0.0.1")
    monkeypatch.setenv("CHROMADB_PORT", str(_free_port()))
    with pytest.raises(RuntimeError):
        cc.get_chroma_client()
    # A failed connection must leave the singleton unset so a later call
    # (once ChromaDB is up) can succeed.
    assert cc._client is None


def test_chroma_http_client_v2_collection_lifecycle_and_numpy_serialization():
    calls = []
    store = {
        "col-1": {
            "id": "col-1",
            "name": "odysseus_rag_fastembed",
            "metadata": {"embedding_lane": "fastembed", "embedding_dimension": 4},
            "rows": {},
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content.decode("utf-8")) if request.content else None
        calls.append((request.method, path, body))

        if path == "/api/v2/heartbeat":
            return httpx.Response(200, json={"nanosecond heartbeat": 123456789})

        base = "/api/v2/tenants/default_tenant/databases/default_database/collections"
        if request.method == "POST" and path == base:
            return httpx.Response(200, json={
                "id": "col-1",
                "name": body["name"],
                "metadata": body.get("metadata") or {},
            })
        if request.method == "GET" and path == f"{base}/odysseus_rag_fastembed":
            c = store["col-1"]
            return httpx.Response(200, json={"id": c["id"], "name": c["name"], "metadata": c["metadata"]})
        if request.method == "DELETE" and path == f"{base}/odysseus_rag_fastembed":
            return httpx.Response(200, json={})
        if path == f"{base}/col-1/count":
            return httpx.Response(200, json=len(store["col-1"]["rows"]))
        if path in (f"{base}/col-1/add", f"{base}/col-1/upsert"):
            for i, row_id in enumerate(body["ids"]):
                store["col-1"]["rows"][row_id] = {
                    "embedding": body["embeddings"][i],
                    "document": (body.get("documents") or [None] * len(body["ids"]))[i],
                    "metadata": (body.get("metadatas") or [{}] * len(body["ids"]))[i],
                }
            return httpx.Response(201, json=True)
        if path == f"{base}/col-1/update":
            for i, row_id in enumerate(body["ids"]):
                if row_id in store["col-1"]["rows"]:
                    store["col-1"]["rows"][row_id]["metadata"] = body["metadatas"][i]
            return httpx.Response(200, json=True)
        if path == f"{base}/col-1/get":
            rows = store["col-1"]["rows"]
            ids = [r for r in (body.get("ids") or list(rows.keys())) if r in rows]
            return httpx.Response(200, json={
                "ids": ids,
                "documents": [rows[r]["document"] for r in ids],
                "metadatas": [rows[r]["metadata"] for r in ids],
                "embeddings": [rows[r]["embedding"] for r in ids],
            })
        if path == f"{base}/col-1/query":
            rows = store["col-1"]["rows"]
            ids = list(rows.keys())[: body["n_results"]]
            return httpx.Response(200, json={
                "ids": [ids],
                "documents": [[rows[r]["document"] for r in ids]],
                "metadatas": [[rows[r]["metadata"] for r in ids]],
                "distances": [[0.12 for _ in ids]],
            })
        if path == f"{base}/col-1/delete":
            for row_id in body.get("ids") or []:
                store["col-1"]["rows"].pop(row_id, None)
            return httpx.Response(200, json={"deleted": 1})
        return httpx.Response(404, json={"error": "NotFound", "message": path})

    client = cc.ChromaHttpClient(host="localhost", port=8100, transport=httpx.MockTransport(handler))
    assert client.heartbeat() == 123456789
    assert client.api_version == "v2"

    col = client.get_or_create_collection("odysseus_rag_fastembed", metadata={"embedding_lane": "fastembed"})
    assert col.id == "col-1"
    assert col.name == "odysseus_rag_fastembed"
    assert col.count() == 0

    # Pass numpy ndarray embeddings to verify _to_jsonable handles ndarrays without orjson
    emb = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    col.add(ids=["doc_1"], embeddings=emb, documents=["hello harlem"], metadatas=[{"owner": "alice"}])
    assert col.count() == 1

    got = col.get(ids=["doc_1"], include=["documents", "metadatas", "embeddings"])
    assert got["ids"] == ["doc_1"]
    assert got["documents"] == ["hello harlem"]
    assert got["metadatas"] == [{"owner": "alice"}]

    col.update(ids=["doc_1"], metadatas=[{"owner": "bob"}])
    queried = col.query(query_embeddings=emb, n_results=1, include=["documents", "metadatas", "distances"])
    assert queried["ids"] == [["doc_1"]]
    assert queried["metadatas"] == [[{"owner": "bob"}]]
    assert queried["distances"] == [[0.12]]

    col.upsert(ids=["doc_1"], embeddings=emb, documents=["updated doc"], metadatas=[{"owner": "bob"}])
    col.delete(ids=["doc_1"])
    assert col.count() == 0


def test_chroma_http_client_falls_back_to_v1_when_v2_returns_404():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/heartbeat":
            return httpx.Response(404, json={"detail": "Not Found"})
        if request.url.path == "/api/v1/heartbeat":
            return httpx.Response(200, json={"nanosecond heartbeat": 42})
        if request.url.path == "/api/v1/collections/legacy_col":
            return httpx.Response(200, json={"id": "uuid-v1", "name": "legacy_col", "metadata": {}})
        return httpx.Response(404)

    client = cc.ChromaHttpClient(host="localhost", port=8100, transport=httpx.MockTransport(handler))
    assert client.heartbeat() == 42
    assert client.api_version == "v1"
    col = client.get_collection("legacy_col")
    assert col.id == "uuid-v1"
    assert col.name == "legacy_col"
