from __future__ import annotations
import json
from typing import Any, Dict

from ..providers.notion_provider import NotionProvider
from ..store import STORE

NOTION = NotionProvider()

def add_notion_todo(payload: str) -> str:
    """
    Add task to to-do in the Notion “Tasks” page.

    Args:
        payload: str
            Either a plain task title (string), or a JSON object string like
            '{"text": "...", "thread_id": "..."}'. "thread_id" is optional.

    Returns:
        str: The created Notion block_id.

    Notes:
        If "thread_id" is provided, the email thread is linked to the Notion
        block in the tiny store.
    """

    # Debugging line
    print("[add_notion_todo] invoked")

    thread_id = None
    task_text = payload

    # Allow JSON payload with optional thread_id
    try:
        obj = json.loads(payload)
        if isinstance(obj, dict):
            task_text = str(obj.get("text", "")).strip() or payload
            # accept either "thread_id" or a more explicit "email_thread_id"
            thread_id = obj.get("thread_id") or obj.get("email_thread_id")
    except Exception:
        # payload was a plain string; that's fine
        pass

    block_id = NOTION.add_todo(task_text)

    # Link mapping if we know the originating thread
    if thread_id:
        try:
            STORE.upsert_mapping(thread_id=thread_id, notion_block_id=block_id)
        except Exception:
            # Don't break tool flow on store errors
            pass

    return block_id


def list_unchecked_tasks(_: str = "") -> str:
    """
    List all open (unchecked) to-dos from the Notion “Tasks” page.

    Returns:
        JSON array string: '[{"block_id": str, "text": str}]'
    """

    # Debugging line
    print("[list_unchecked_tasks] invoked")

    tasks = NOTION.list_unchecked()
    items = [
        {"block_id": t.get("block_id", t.get("id")), "text": t["text"]}
        for t in tasks
    ]
    return json.dumps(items, ensure_ascii=False)
