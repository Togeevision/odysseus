import base64
import importlib.util
import json
from pathlib import Path
import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "archive_search.py"
SPEC = importlib.util.spec_from_file_location("archive_search", MODULE_PATH)
archive_search = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive_search)


def test_generate_scoped_search_key_embeds_hmac_prefix_and_filter_params():
    parent_key = "search_only_parent_key_123456"
    params = {"filter_by": "cityKey:=`montreal` && entityId:=`la-sala-rossa`", "per_page": 10}

    scoped = archive_search.generate_scoped_search_key(parent_key, params)
    decoded = base64.b64decode(scoped.encode("utf-8")).decode("utf-8")

    assert parent_key[:4] in decoded
    assert '"filter_by":"cityKey:=`montreal` && entityId:=`la-sala-rossa`"' in decoded


def test_search_cultural_archive_rejects_bootstrap_admin_key(monkeypatch):
    monkeypatch.setenv("TYPESENSE_API_KEY", "same_bootstrap_secret_key")
    monkeypatch.setenv("TYPESENSE_SEARCH_ONLY_KEY", "same_bootstrap_secret_key")

    with pytest.raises(ValueError, match="not the bootstrap TYPESENSE_API_KEY"):
        archive_search.search_cultural_archive(query="jazz", city_key="montreal")


def test_search_cultural_archive_queries_with_scoped_key_and_truncates_snippets(monkeypatch):
    captured = {}

    def fake_request(base_url, api_key, path, *, method="GET", body=None):
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        captured["path"] = path
        return (
            200,
            json.dumps(
                {
                    "found": 1,
                    "hits": [
                        {
                            "document": {
                                "id": "snap_1",
                                "title": "Casa del Popolo Lineup",
                                "url": "https://casadelpopolo.com/events",
                                "domain": "casadelpopolo.com",
                                "timestamp": "1761912000",
                                "archived_at": 1761912000,
                                "cityKey": "montreal",
                                "entityType": "venue",
                                "entityId": "casa-del-popolo",
                                "watchId": "watch_1",
                                "contentHash": "a" * 64,
                                "archivePath": "/archive/1761912000/index.html",
                                "content": "x" * 1200,
                            }
                        }
                    ],
                    "facet_counts": [
                        {"field_name": "cityKey", "counts": [{"value": "montreal", "count": 1}]}
                    ],
                }
            ),
        )

    monkeypatch.delenv("TYPESENSE_API_KEY", raising=False)
    monkeypatch.setattr(archive_search, "_request_json", fake_request)

    result = archive_search.search_cultural_archive(
        query="lineup",
        city_key="montreal",
        entity_id="casa-del-popolo",
        limit=5,
        base_url="http://127.0.0.1:8108",
        search_only_key="ro_search_key_abcdef1234",
    )

    assert captured["api_key"] != "ro_search_key_abcdef1234"
    decoded_key = base64.b64decode(captured["api_key"].encode("utf-8")).decode("utf-8")
    assert "cityKey:=`montreal` && entityId:=`casa-del-popolo`" in decoded_key
    assert "filter_by=" in captured["path"]
    assert result["found"] == 1
    assert len(result["hits"][0]["snippet"]) == 600
    assert result["facets"]["cityKey"] == [{"value": "montreal", "count": 1}]


def test_mint_search_only_key_requests_documents_search_scope(monkeypatch):
    captured = {}

    def fake_request(base_url, api_key, path, *, method="GET", body=None):
        captured["method"] = method
        captured["path"] = path
        captured["body"] = json.loads(body.decode("utf-8"))
        return 201, json.dumps({"id": 7, "value": "minted_search_only_key_xyz"})

    monkeypatch.setattr(archive_search, "_request_json", fake_request)

    minted = archive_search.mint_search_only_key("http://127.0.0.1:8108", "bootstrap_admin_key")

    assert minted == "minted_search_only_key_xyz"
    assert captured["method"] == "POST"
    assert captured["path"] == "/keys"
    assert captured["body"]["actions"] == ["documents:search"]
    assert captured["body"]["collections"] == ["cultural_archive"]
