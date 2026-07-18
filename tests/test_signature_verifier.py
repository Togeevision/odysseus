import os
import json
import time
import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.database import Base, UsedNonce
import core.database
import core.signature_verifier
from core.signature_verifier import require_togee_signature, build_canonical_request

class MockUrl:
    def __init__(self, path, query):
        self.path = path
        self.query = query

class MockRequest:
    def __init__(self, method, path, query, headers, body_bytes):
        self.method = method
        self.url = MockUrl(path, query)
        self.headers = headers
        self._body_bytes = body_bytes

    async def body(self):
        return self._body_bytes

# Load shared test vector
VECTOR_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../security/test-vectors/odysseus-hmac-v1.json"
)
with open(VECTOR_PATH, "r") as f:
    vector = json.load(f)

@pytest.fixture(autouse=True)
def mock_db_session(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    
    from contextlib import contextmanager
    @contextmanager
    def mock_get_session():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(core.database, "get_db_session", mock_get_session)
    monkeypatch.setattr(core.signature_verifier, "get_db_session", mock_get_session)
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("ODYSSEUS_INTERNAL_TOKEN", vector["secret"])

@pytest.mark.asyncio
async def test_signature_verifier_vector_success():
    req = MockRequest(
        method=vector["method"],
        path=vector["path"],
        query=vector["query"],
        headers={
            "X-Togee-Signature-Version": vector["version"],
            "X-Togee-Signature-Timestamp": vector["timestamp"],
            "X-Togee-Signature-Nonce": vector["nonce"],
            "X-Togee-Signature-Caller": vector["caller"],
            "X-Togee-Signature-Audience": vector["audience"],
            "X-Togee-Body-SHA256": vector["bodySha256"],
            "X-Togee-Signature": vector["signature"],
            "content-type": vector["contentType"]
        },
        body_bytes=vector["bodyUtf8"].encode("utf-8")
    )

    from unittest.mock import patch
    with patch("time.time", return_value=int(vector["timestamp"])):
        assert await require_togee_signature(req) is True

@pytest.mark.asyncio
async def test_signature_verifier_missing_headers():
    req = MockRequest(
        method="POST",
        path="/api/research/jobs",
        query="",
        headers={},
        body_bytes=b""
    )
    with pytest.raises(HTTPException) as exc:
        await require_togee_signature(req)
    assert exc.value.status_code == 401
    assert exc.value.detail == "SIGNATURE_REQUIRED"

@pytest.mark.asyncio
async def test_signature_verifier_version_unsupported():
    headers = {
        "X-Togee-Signature-Version": "v2",
        "X-Togee-Signature-Timestamp": str(int(time.time())),
        "X-Togee-Signature-Nonce": "550e8400-e29b-41d4-a716-446655440000",
        "X-Togee-Signature-Caller": "research-worker",
        "X-Togee-Signature-Audience": "odysseus-research",
        "X-Togee-Body-SHA256": "abc",
        "X-Togee-Signature": "sig"
    }
    req = MockRequest("POST", "/api/research/jobs", "", headers, b"")
    with pytest.raises(HTTPException) as exc:
        await require_togee_signature(req)
    assert exc.value.status_code == 426
    assert exc.value.detail == "SIGNATURE_VERSION_UNSUPPORTED"

@pytest.mark.asyncio
async def test_signature_verifier_malformed_timestamp():
    headers = {
        "X-Togee-Signature-Version": "v1",
        "X-Togee-Signature-Timestamp": "not-a-number",
        "X-Togee-Signature-Nonce": "550e8400-e29b-41d4-a716-446655440000",
        "X-Togee-Signature-Caller": "research-worker",
        "X-Togee-Signature-Audience": "odysseus-research",
        "X-Togee-Body-SHA256": "abc",
        "X-Togee-Signature": "sig"
    }
    req = MockRequest("POST", "/api/research/jobs", "", headers, b"")
    with pytest.raises(HTTPException) as exc:
        await require_togee_signature(req)
    assert exc.value.status_code == 400
    assert exc.value.detail == "MALFORMED_SIGNATURE_REQUEST"

@pytest.mark.asyncio
async def test_signature_verifier_expired_timestamp():
    expired_time = int(time.time()) - 400
    headers = {
        "X-Togee-Signature-Version": "v1",
        "X-Togee-Signature-Timestamp": str(expired_time),
        "X-Togee-Signature-Nonce": "550e8400-e29b-41d4-a716-446655440000",
        "X-Togee-Signature-Caller": "research-worker",
        "X-Togee-Signature-Audience": "odysseus-research",
        "X-Togee-Body-SHA256": "abc",
        "X-Togee-Signature": "sig"
    }
    req = MockRequest("POST", "/api/research/jobs", "", headers, b"")
    with pytest.raises(HTTPException) as exc:
        await require_togee_signature(req)
    assert exc.value.status_code == 401
    assert exc.value.detail == "SIGNATURE_EXPIRED"

@pytest.mark.asyncio
async def test_signature_verifier_malformed_nonce():
    headers = {
        "X-Togee-Signature-Version": "v1",
        "X-Togee-Signature-Timestamp": str(int(time.time())),
        "X-Togee-Signature-Nonce": "invalid-nonce-shape!",
        "X-Togee-Signature-Caller": "research-worker",
        "X-Togee-Signature-Audience": "odysseus-research",
        "X-Togee-Body-SHA256": "abc",
        "X-Togee-Signature": "sig"
    }
    req = MockRequest("POST", "/api/research/jobs", "", headers, b"")
    with pytest.raises(HTTPException) as exc:
        await require_togee_signature(req)
    assert exc.value.status_code == 400
    assert exc.value.detail == "MALFORMED_SIGNATURE_REQUEST"

def test_signature_verifier_path_validation():
    # Verify build_canonical_request rejects double slashes, dot segments, and percent-encoded dot segments
    with pytest.raises(ValueError, match="Path contains invalid dot segments or duplicate slashes"):
        build_canonical_request("GET", "/api//research", "", "123", "abc", "caller", "aud", "type", "hash")

    with pytest.raises(ValueError, match="Path contains prohibited dot segments"):
        build_canonical_request("GET", "/api/research/jobs/.", "", "123", "abc", "caller", "aud", "type", "hash")

    with pytest.raises(ValueError, match="Path contains prohibited dot segments"):
        build_canonical_request("GET", "/api/research/jobs/..", "", "123", "abc", "caller", "aud", "type", "hash")

    with pytest.raises(ValueError, match="Path contains prohibited dot segments"):
        build_canonical_request("GET", "/api/research/jobs/%2E", "", "123", "abc", "caller", "aud", "type", "hash")

    with pytest.raises(ValueError, match="Path contains prohibited dot segments"):
        build_canonical_request("GET", "/api/research/jobs/%2E%2E", "", "123", "abc", "caller", "aud", "type", "hash")

def test_signature_verifier_query_normalization():
    from core.signature_verifier import normalize_query
    # 1. Sort by key then value
    assert normalize_query("b=2&a=1") == "a=1&b=2"
    # 2. Preserve duplicates
    assert normalize_query("a=2&a=1") == "a=1&a=2"
    # 3. Percent-encode space as %20
    assert normalize_query("q=hello world") == "q=hello%20world"
    assert normalize_query("q=hello+world") == "q=hello%20world"
    # 4. Unicode values
    assert normalize_query("city=Montréal") == "city=Montr%C3%A9al"
    # 5. Ordinal sorting including A, a, %, ~, é, %E2%82%AC, duplicates, and empty keys/values
    assert normalize_query("~=1&%=2&a=3&A=4") == "%25=2&A=4&a=3&~=1"
    assert normalize_query("key=value&key=") == "key=&key=value"
    assert normalize_query("=value&=") == "=&=value"

@pytest.mark.asyncio
async def test_signature_verifier_disallowed_route():
    headers = {
        "X-Togee-Signature-Version": "v1",
        "X-Togee-Signature-Timestamp": str(int(time.time())),
        "X-Togee-Signature-Nonce": "550e8400-e29b-41d4-a716-446655440000",
        "X-Togee-Signature-Caller": "research-worker",
        "X-Togee-Signature-Audience": "odysseus-research",
        "X-Togee-Body-SHA256": "abc",
        "X-Togee-Signature": "sig"
    }
    req = MockRequest("POST", "/api/research/admin/wipe", "", headers, b"")
    with pytest.raises(HTTPException) as exc:
        await require_togee_signature(req)
    assert exc.value.status_code == 403
    assert exc.value.detail == "ROUTE_NOT_ALLOWED"

@pytest.mark.asyncio
async def test_signature_verifier_nonce_replay():
    import hmac
    import hashlib
    now_ts = int(time.time())
    new_nonce = "770e8400-e29b-41d4-a716-44665544aaaa"
    
    canonical_req = build_canonical_request(
        method=vector["method"],
        path=vector["path"],
        query=vector["query"],
        timestamp=str(now_ts),
        nonce=new_nonce,
        caller=vector["caller"],
        audience=vector["audience"],
        content_type=vector["contentType"],
        body_sha256=vector["bodySha256"]
    )
    
    signature = hmac.new(
        vector["secret"].encode("utf-8"),
        canonical_req.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    req = MockRequest(
        method=vector["method"],
        path=vector["path"],
        query=vector["query"],
        headers={
            "X-Togee-Signature-Version": vector["version"],
            "X-Togee-Signature-Timestamp": str(now_ts),
            "X-Togee-Signature-Nonce": new_nonce,
            "X-Togee-Signature-Caller": vector["caller"],
            "X-Togee-Signature-Audience": vector["audience"],
            "X-Togee-Body-SHA256": vector["bodySha256"],
            "X-Togee-Signature": signature,
            "content-type": vector["contentType"]
        },
        body_bytes=vector["bodyUtf8"].encode("utf-8")
    )

    # First execution succeeds
    assert await require_togee_signature(req) is True

    # Second execution with same nonce throws 409 NONCE_REPLAYED
    with pytest.raises(HTTPException) as exc:
        await require_togee_signature(req)
    assert exc.value.status_code == 409
    assert exc.value.detail == "NONCE_REPLAYED"

@pytest.mark.asyncio
async def test_signature_verifier_expired_nonce_cleanup():
    # Insert an expired nonce manually
    with core.database.get_db_session() as db:
        expired = UsedNonce(
            nonce="expired-nonce",
            caller="research-worker",
            request_hash="abc",
            expires_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=1)
        )
        db.add(expired)
        db.commit()

        assert db.query(UsedNonce).filter(UsedNonce.nonce == "expired-nonce").count() == 1

    # Run any signature validation (using a different nonce so signature verification itself passes)
    # We will generate a new valid nonce and signature to trigger the pipeline
    import hmac
    import hashlib
    now_ts = int(vector["timestamp"])
    new_nonce = "660e8400-e29b-41d4-a716-446655449999"
    
    canonical_req = build_canonical_request(
        method=vector["method"],
        path=vector["path"],
        query=vector["query"],
        timestamp=str(now_ts),
        nonce=new_nonce,
        caller=vector["caller"],
        audience=vector["audience"],
        content_type=vector["contentType"],
        body_sha256=vector["bodySha256"]
    )
    
    signature = hmac.new(
        vector["secret"].encode("utf-8"),
        canonical_req.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    req = MockRequest(
        method=vector["method"],
        path=vector["path"],
        query=vector["query"],
        headers={
            "X-Togee-Signature-Version": vector["version"],
            "X-Togee-Signature-Timestamp": str(now_ts),
            "X-Togee-Signature-Nonce": new_nonce,
            "X-Togee-Signature-Caller": vector["caller"],
            "X-Togee-Signature-Audience": vector["audience"],
            "X-Togee-Body-SHA256": vector["bodySha256"],
            "X-Togee-Signature": signature,
            "content-type": vector["contentType"]
        },
        body_bytes=vector["bodyUtf8"].encode("utf-8")
    )

    from unittest.mock import patch
    with patch("time.time", return_value=now_ts):
        await require_togee_signature(req)

    # Check that expired nonce has been cleaned up
    with core.database.get_db_session() as db:
        assert db.query(UsedNonce).filter(UsedNonce.nonce == "expired-nonce").count() == 0
