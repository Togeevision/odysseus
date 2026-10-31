import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "archivebox_typesense_indexer.py"
SPEC = importlib.util.spec_from_file_location("archivebox_typesense_indexer", MODULE_PATH)
indexer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(indexer)


def test_build_document_extracts_bounded_text_and_explicit_facets(tmp_path):
    snapshot = tmp_path / "archive" / "20261031123000"
    (snapshot / "readability").mkdir(parents=True)
    (snapshot / "index.json").write_text(
        json.dumps(
            {
                "url": "https://example.org/room/signal",
                "title": "Room Signal",
                "timestamp": "20261031123000",
                "tags_str": "city:rochester,entity-type:venue,entity:hallock,watch:watch-7,freeform",
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "readability" / "content.txt").write_text("signal " * 40, encoding="utf-8")

    document = indexer.build_document(snapshot / "index.json", tmp_path / "archive", 50)

    assert len(document["content"]) == 50
    assert document["cityKey"] == "rochester"
    assert document["entityType"] == "venue"
    assert document["entityId"] == "hallock"
    assert document["watchId"] == "watch-7"
    assert document["domain"] == "example.org"
    assert document["archivePath"] == "/archive/20261031123000/index.html"
    assert len(document["contentHash"]) == 64


def test_build_document_supports_canonical_nested_archive_tree(tmp_path):
    snapshot = tmp_path / "archive" / "users" / "operator" / "snapshots" / "2026-10-31" / "example.org" / "uuid-1"
    snapshot.mkdir(parents=True)
    (snapshot / "index.json").write_text(
        json.dumps({"url": "https://example.org/a", "timestamp": "1761912000", "tags": [{"name": "city:montreal"}]}),
        encoding="utf-8",
    )

    files = list(indexer.iter_index_files(tmp_path))
    document = indexer.build_document(files[0], tmp_path / "archive", 100)

    assert document["cityKey"] == "montreal"
    assert document["archivePath"].endswith("/uuid-1/index.html")


def test_document_id_is_idempotent(tmp_path):
    snapshot = tmp_path / "archive" / "1761912000"
    snapshot.mkdir(parents=True)
    index_file = snapshot / "index.json"
    index_file.write_text(json.dumps({"url": "https://example.org/a", "timestamp": "1761912000"}), encoding="utf-8")

    first = indexer.build_document(index_file, tmp_path / "archive", 100)
    second = indexer.build_document(index_file, tmp_path / "archive", 100)

    assert first["id"] == second["id"]


def test_import_documents_counts_partial_failures(monkeypatch):
    monkeypatch.setattr(
        indexer,
        "request_json",
        lambda *args, **kwargs: (200, '{"success":true}\n{"success":false,"error":"bad"}'),
    )

    assert indexer.import_documents("http://typesense:8108", "secret", [{"id": "1"}, {"id": "2"}]) == (1, 1)
