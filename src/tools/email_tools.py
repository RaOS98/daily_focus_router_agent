from __future__ import annotations
import json
from langchain.tools import tool

from ..providers.gmail_provider import GmailProvider
from ..store import STORE

GMAIL = GmailProvider()

@tool("fetch_recent_emails", return_direct=False)
def fetch_recent_emails() -> str:
    """Fetch last 24h emails (subject + snippet). Returns JSON list with keys:
    thread_id, subject, snippet, sender, date. Sorted newest-first.
    Call at most once per run."""

    # Debugging line
    print("[fetch_recent_emails] invoked")

    raw = GMAIL.fetch_last_24h()

    # sort newest-first and trim
    raw = sorted(raw, key=lambda m: m.get("date", ""), reverse=True)
    out = []
    for m in raw:
        item = {
            "thread_id": m.get("thread_id"),
            "subject": (m.get("subject") or "")[:200],
            "snippet": (m.get("snippet") or "")[:300],
            "sender": m.get("from"),
            "date": m.get("date"),
        }
        out.append(item)

        # Seed a mapping row so later tools can attach Notion/Calendar IDs.
        tid = item.get("thread_id")
        if tid:
            STORE.upsert_mapping(thread_id=tid)

    # Optional: if your Gmail provider exposes a sync cursor/history id, persist it.
    # Example (safe no-op if attribute/method doesn't exist):
    cursor = getattr(GMAIL, "last_history_id", None)
    if cursor:
        try:
            STORE.set_cursor("gmail", str(cursor))
        except Exception:
            pass

    return json.dumps(out, ensure_ascii=False)
