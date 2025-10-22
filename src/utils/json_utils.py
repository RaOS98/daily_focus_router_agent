# src/utils/json_utils.py
from __future__ import annotations
import json
from typing import Any, List

def extract_json_array(s: str) -> List[Any]:
    """Best-effort: pull a top-level JSON array from an arbitrary string."""
    if not isinstance(s, str):
        return []
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except Exception:
            return []
    return []
