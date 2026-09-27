"""Bien trang thai runtime dung chung trong FastAPI app.

Module nay gom cache ngan han, lock ingest va bucket rate-limit. Tach rieng de
schema chi con khai bao request model, khong chua state van hanh.
"""

import threading


response_cache: dict[str, str] = {}
INGEST_LOCK = threading.Lock()
rate_limit_buckets: dict[str, list[float]] = {}
