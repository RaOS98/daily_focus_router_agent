from __future__ import annotations
import json
from langchain.tools import tool

from ..providers.gmail_provider import GmailProvider

GMAIL = GmailProvider()

@tool("fetch_recent_emails", return_direct=False)
def fetch_recent_emails(_: str = "") -> str:
    """Fetch last 48h emails (subject + snippet). Input is ignored. Returns JSON list with keys: thread_id, subject, snippet, sender, date."""
    raw = GMAIL.fetch_last_48h()
    return json.dumps(raw, ensure_ascii=False)
