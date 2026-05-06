from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_hash(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, Path):
        payload = value.read_bytes()
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = stable_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def short_hash(value: Any, length: int = 12) -> str:
    return content_hash(value)[:length]
