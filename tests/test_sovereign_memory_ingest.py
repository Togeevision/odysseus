from types import SimpleNamespace
from unittest.mock import MagicMock

from routes.memory.memory_routes import AddSovereignRequest, setup_memory_routes


def _route(router, path, method):
    for route in router.routes:
        if route.path == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(path)


def test_add_sovereign_preserves_raw_text_and_stores_routing_metadata():
    memory_manager = MagicMock()
    memory_manager.add_entry.return_value = {
        "id": "memory-1",
        "text": "Chronicle body",
        "owner": "admin",
    }
    memory_manager.load_all.return_value = []
    memory_vector = MagicMock(healthy=True)
    router = setup_memory_routes(memory_manager, SimpleNamespace(), memory_vector)
    add_sovereign = _route(router, "/api/memory/add-sovereign", "POST")

    result = add_sovereign(AddSovereignRequest(
        databaseKey="chronicle:abc",
        text="Chronicle body",
        caption="Chronicle body",
        location="montreal",
    ))

    memory_manager.add_entry.assert_called_once_with(
        text="Chronicle body",
        source="sovereign-dispatch",
        category="fact",
        owner="admin",
    )
    saved_entry = memory_manager.save.call_args.args[0][0]
    assert saved_entry["metadata"] == {
        "database_key": "chronicle:abc",
        "location": "montreal",
        "text_shape": "raw",
    }
    memory_vector.add.assert_called_once_with("memory-1", "Chronicle body")
    assert result["text"] == "Chronicle body"
    assert result["count"] == 1


def test_add_sovereign_keeps_legacy_caption_shape_for_old_callers():
    memory_manager = MagicMock()
    memory_manager.add_entry.return_value = {"id": "memory-2", "owner": "admin"}
    memory_manager.load_all.return_value = []
    router = setup_memory_routes(memory_manager, SimpleNamespace())
    add_sovereign = _route(router, "/api/memory/add-sovereign", "POST")

    result = add_sovereign(AddSovereignRequest(
        databaseKey="dispatch:abc",
        caption="Legacy caption",
        location="nyc",
    ))

    assert result["text"] == "[dispatch:abc] Location: nyc — Legacy caption"
    assert memory_manager.save.call_args.args[0][0]["metadata"]["text_shape"] == "legacy-caption"
