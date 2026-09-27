"""Rate-limit helpers cho cac endpoint FastAPI."""

import time

from fastapi import Request
from fastapi.responses import JSONResponse

from botai.core.config import RATE_LIMIT_RULES
from botai.core.state import rate_limit_buckets

__all__ = ["_client_ip", "_rate_limit", "_rate_limit_json"]

def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limit(request: Request, rule_name: str) -> tuple[bool, str]:
    max_requests, window_seconds = RATE_LIMIT_RULES[rule_name]
    now = time.monotonic()
    key = f"{rule_name}:{_client_ip(request)}"
    bucket = [
        timestamp for timestamp in rate_limit_buckets.get(key, [])
        if now - timestamp < window_seconds
    ]
    if len(bucket) >= max_requests:
        retry_after = max(1, int(window_seconds - (now - bucket[0])))
        rate_limit_buckets[key] = bucket
        return False, f"Bạn gửi quá nhiều yêu cầu. Vui lòng thử lại sau {retry_after} giây."
    bucket.append(now)
    rate_limit_buckets[key] = bucket
    return True, ""


def _rate_limit_json(request: Request, rule_name: str):
    ok, message = _rate_limit(request, rule_name)
    if ok:
        return None
    return JSONResponse(
        {"ok": False, "error": "rate_limited", "message": message},
        status_code=429,
    )
