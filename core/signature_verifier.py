import os
import hmac
import hashlib
import time
import uuid
import urllib.parse
from datetime import datetime, timezone, timedelta
from fastapi import Request, HTTPException, status, Depends
from sqlalchemy.exc import IntegrityError
from core.database import get_db_session, UsedNonce, utcnow_naive

CALLER_POLICIES = {
    "research-worker": {
        "audience": "odysseus-research",
        "methods": {"POST", "GET"},
        "paths": {
            "/api/research/jobs",
            "/api/research/jobs/status",
            "/api/research/jobs/cancel",
            "/api/memory/find-similar",
            "/api/memory/add-sovereign",
        },
    },
}

def normalize_content_type(content_type: str) -> str:
    if not content_type:
        return ""
    return content_type.lower().split(";")[0].strip()

def normalize_query(query_str: str) -> str:
    if not query_str:
        return ""
    pairs = urllib.parse.parse_qsl(query_str, keep_blank_values=True)
    
    def rfc3986(s: str) -> str:
        return urllib.parse.quote(s, safe="~")\
            .replace("!", "%21")\
            .replace("'", "%27")\
            .replace("(", "%28")\
            .replace(")", "%29")\
            .replace("*", "%2A")

    encoded_pairs = [(rfc3986(k), rfc3986(v)) for k, v in pairs]
    encoded_pairs.sort(key=lambda x: (x[0], x[1]))
    return "&".join(f"{k}={v}" for k, v in encoded_pairs)

def build_canonical_request(
    method: str,
    path: str,
    query: str,
    timestamp: str,
    nonce: str,
    caller: str,
    audience: str,
    content_type: str,
    body_sha256: str
) -> str:
    norm_method = method.upper()
    norm_path = path.strip()
    
    if not norm_path.startswith('/'):
        raise ValueError("Path must begin with a single slash")
    if '//' in norm_path:
        raise ValueError("Path contains invalid dot segments or duplicate slashes")

    segments = norm_path.split('/')
    for segment in segments:
        if "%" in segment:
            import re
            for match in re.finditer(r'%(..)?', segment):
                m = match.group(0)
                if len(m) < 3 or not all(c in '0123456789abcdefABCDEF' for c in m[1:]):
                    raise ValueError("Path contains malformed percent encoding")
        
        decoded = urllib.parse.unquote(segment)
        if segment in ('.', '..') or decoded in ('.', '..'):
            raise ValueError("Path contains prohibited dot segments")

    norm_query = normalize_query(query)
    norm_content_type = normalize_content_type(content_type)

    fields = [
        "v1",
        norm_method,
        norm_path,
        norm_query,
        str(timestamp),
        str(nonce),
        str(caller),
        str(audience),
        norm_content_type,
        str(body_sha256)
    ]
    return "\n".join(fields)

async def require_togee_signature(request: Request):
    headers = request.headers
    version = headers.get("X-Togee-Signature-Version")
    timestamp_str = headers.get("X-Togee-Signature-Timestamp")
    nonce = headers.get("X-Togee-Signature-Nonce")
    caller = headers.get("X-Togee-Signature-Caller")
    audience = headers.get("X-Togee-Signature-Audience")
    signature = headers.get("X-Togee-Signature")
    body_sha256_header = headers.get("X-Togee-Body-SHA256")

    if not all([version, timestamp_str, nonce, caller, audience, signature, body_sha256_header]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="SIGNATURE_REQUIRED"
        )

    if version != "v1":
        raise HTTPException(
            status_code=status.HTTP_426_UPGRADE_REQUIRED,
            detail="SIGNATURE_VERSION_UNSUPPORTED"
        )

    try:
        if not timestamp_str.isdigit():
            raise ValueError()
        timestamp = int(timestamp_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MALFORMED_SIGNATURE_REQUEST"
        )

    now = int(time.time())
    if abs(now - timestamp) > 300:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="SIGNATURE_EXPIRED"
        )

    clean_nonce = nonce.lower().strip()
    is_valid_uuid = False
    try:
        uuid.UUID(clean_nonce)
        is_valid_uuid = True
    except ValueError:
        pass
    
    is_valid_hex = len(clean_nonce) == 32 and all(c in "0123456789abcdef" for c in clean_nonce)
    
    if not (is_valid_uuid or is_valid_hex):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MALFORMED_SIGNATURE_REQUEST"
        )

    if not caller:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CALLER_NOT_ALLOWED"
        )
    
    policy = CALLER_POLICIES.get(caller)
    if not policy:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CALLER_NOT_ALLOWED"
        )

    if audience != policy["audience"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="AUDIENCE_MISMATCH"
        )

    method = request.method.upper()
    path = request.url.path

    if method not in policy["methods"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="ROUTE_NOT_ALLOWED"
        )

    if path not in policy["paths"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="ROUTE_NOT_ALLOWED"
        )

    body_bytes = await request.body()
    computed_body_sha = hashlib.sha256(body_bytes).hexdigest()
    if computed_body_sha != body_sha256_header.lower().strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="SIGNATURE_INVALID"
        )

    hmac_key = os.getenv("ODYSSEUS_INTERNAL_TOKEN")
    if not hmac_key:
        if os.getenv("TESTING") == "true":
            hmac_key = "test-only-secret"
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="HMAC secret key is not configured"
            )

    try:
        canonical_req = build_canonical_request(
            method=method,
            path=path,
            query=request.url.query,
            timestamp=timestamp_str,
            nonce=clean_nonce,
            caller=caller,
            audience=audience,
            content_type=headers.get("content-type", ""),
            body_sha256=computed_body_sha
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MALFORMED_SIGNATURE_REQUEST"
        )

    expected_sig = hmac.new(
        hmac_key.encode("utf-8"),
        canonical_req.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(signature.lower().strip(), expected_sig):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="SIGNATURE_INVALID"
        )

    expires_at = datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(tzinfo=None) + timedelta(seconds=900)
    
    with get_db_session() as db:
        # Nonce cleanup
        try:
            db.query(UsedNonce).filter(UsedNonce.expires_at < utcnow_naive()).delete()
            db.commit()
        except Exception:
            db.rollback()

        # Commit used nonce
        try:
            used = UsedNonce(
                nonce=clean_nonce,
                caller=caller,
                request_hash=hashlib.sha256(canonical_req.encode("utf-8")).hexdigest(),
                expires_at=expires_at
            )
            db.add(used)
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="NONCE_REPLAYED"
            )

    return True
