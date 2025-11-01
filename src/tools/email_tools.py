# src/tools/email_tools.py
from __future__ import annotations
from typing import List, Dict

from ..providers.gmail_provider import GmailProvider
from ..store import STORE

GMAIL = GmailProvider()

def fetch_recent_emails() -> List[Dict]:
    """
    Fetch the last 24 hours of emails.

    Returns:
        list[dict]: Newest-first items with keys:
            - thread_id: str
            - subject: str
            - snippet: str
            - sender: str
            - date: str (ISO 8601)

    Notes:
        - Takes no arguments.
        - Call at most once per planning run.
    """
    print("[fetch_recent_emails] invoked")

    raw = GMAIL.fetch_last_24h()
    raw = sorted(raw, key=lambda m: m.get("date", ""), reverse=True)

    out: List[Dict] = []
    for m in raw:
        item = {
            "thread_id": m.get("thread_id"),
            "subject": (m.get("subject") or "")[:200],
            "snippet": (m.get("snippet") or "")[:300],
            "sender": m.get("from"),
            "date": m.get("date"),
        }
        out.append(item)

        # Pre-seed a mapping row so later tools can attach Notion/Calendar IDs.
        tid = item.get("thread_id")
        if tid:
            try:
                STORE.upsert_mapping(thread_id=tid)
            except Exception:
                # Don't block the tool on store write issues.
                pass

    # If your Gmail provider exposes a sync cursor/history id, persist it.
    cursor = getattr(GMAIL, "last_history_id", None)
    if cursor:
        try:
            STORE.set_cursor("gmail", str(cursor))
        except Exception:
            pass

    return out
