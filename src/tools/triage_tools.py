# src/tools/triage_tools.py
from __future__ import annotations
import json
from typing import Any, Dict, List
from langchain.tools import tool
from langchain_ollama import ChatOllama

from ..config import OLLAMA_MODEL, OLLAMA_BASE_URL


def _extract_json_array(s: str) -> List[Any]:
    """Return the first top-level JSON array found in the string, else []."""
    if not isinstance(s, str):
        return []
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            return []
    return []


def _normalize_emails_arg(arg: Any) -> List[Dict[str, Any]]:
    """
    Accepts:
      - list of {thread_id, subject, snippet, sender, date}
      - dict with key 'emails' -> list
      - JSON string of either form
    Returns a list of email dicts with 'thread_id' present.
    """
    items: List[Any]
    if isinstance(arg, list):
        items = arg
    elif isinstance(arg, dict):
        items = arg.get("emails", [])
        if not isinstance(items, list):
            items = []
    elif isinstance(arg, str):
        try:
            obj = json.loads(arg)
            if isinstance(obj, list):
                items = obj
            elif isinstance(obj, dict) and isinstance(obj.get("emails"), list):
                items = obj["emails"]
            else:
                items = _extract_json_array(arg)
        except Exception:
            items = _extract_json_array(arg)
    else:
        items = []

    out: List[Dict[str, Any]] = []
    for e in items:
        if not isinstance(e, dict):
            continue
        tid = (e.get("thread_id") or "").strip()
        if not tid:
            continue
        out.append(
            {
                "thread_id": tid,
                "subject": (e.get("subject") or "")[:200],
                "snippet": (e.get("snippet") or "")[:500],
                "sender": e.get("sender") or "",
                "date": e.get("date") or "",
            }
        )
    return out


@tool("call_filtering_agent", return_direct=False)
def call_filtering_agent(emails: Any = None) -> List[Dict[str, Any]]:
    """
    Use the LLM-powered filtering agent to KEEP only actionable, work-related emails.

    Args:
        emails: list | dict | str

    Returns:
        list[dict]: The filtered emails (same shape as input items).
    """

    print("[call_filtering_agent] invoked")

    items = _normalize_emails_arg(emails)
    if not items:
        return []

    # Keep at most a sane window
    items = items[:30]

    # Build messages per latest docs (dict format; no SystemMessage/HumanMessage classes)
    messages = [
        {
            "role": "system",
            "content": (
                "Filter emails like a disciplined executive assistant.\n"
                "- KEEP items that request a decision/approval, deliverable, meeting/coordination, "
                "deadline/payment, or a substantive reply.\n"
                "- DROP FYIs, newsletters, promos, login/security alerts, generic receipts, or anything "
                "without a user action.\n"
                "Return ONLY a JSON array of thread_id strings to KEEP."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                [
                    {
                        "thread_id": e["thread_id"],
                        "subject": e.get("subject", ""),
                        "snippet": e.get("snippet", ""),
                        "sender": e.get("sender", ""),
                        "date": e.get("date", ""),
                    }
                    for e in items
                ],
                ensure_ascii=False,
            ),
        },
    ]

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )

    resp = llm.invoke(messages)
    raw = getattr(resp, "content", "") or str(resp)

    # Expect ["t1", "t2", ...]
    arr = _extract_json_array(raw)
    keep_ids = {tid.strip() for tid in arr if isinstance(tid, str) and tid.strip()}

    filtered = [e for e in items if e["thread_id"] in keep_ids]
    return filtered
