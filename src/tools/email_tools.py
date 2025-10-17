from __future__ import annotations
from langchain.tools import tool

from ..providers.gmail_provider import GmailProvider

GMAIL = GmailProvider()

@tool("fetch_recent_emails", return_direct=False)
def fetch_recent_emails() -> str:
    """Fetch last 48h emails (subject + snippet). Returns JSON list with keys:
    thread_id, subject, snippet, sender, date. Sorted newest-first.
    Call at most once per run."""
    raw = GMAIL.fetch_last_48h()
    # sort newest-first and trim
    raw = sorted(raw, key=lambda m: m.get("date", ""), reverse=True)
    out = []
    for m in raw:
        out.append({
            "thread_id": m.get("thread_id"),
            "subject": (m.get("subject") or "")[:200],
            "snippet": (m.get("snippet") or "")[:300],
            "sender": m.get("from"),
            "date": m.get("date"),
        })
    import json
    return json.dumps(out, ensure_ascii=False)
