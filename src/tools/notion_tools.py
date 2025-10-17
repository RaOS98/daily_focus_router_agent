from __future__ import annotations
import json
from langchain.tools import tool

from ..providers.notion_provider import NotionProvider

NOTION = NotionProvider()

@tool("add_notion_todo", return_direct=False)
def add_notion_todo(task_text: str) -> str:
    """Create a checkbox to-do in Notion 'Tasks' page.
    Input: a concise one-line task text (English). Returns the Notion block_id.
    Call once per actionable email; avoid duplicates."""
    text = (task_text or "").strip()
    if not text:
        # keep it predictable—models sometimes pass empty strings
        raise ValueError("Task text must be non-empty.")
    # keep line short; prevents model from dumping paragraphs into Notion
    if len(text) > 160:
        text = text[:157] + "..."
    block_id = NOTION.add_todo(text)
    return block_id

@tool("list_unchecked_tasks", return_direct=False)
def list_unchecked_tasks() -> str:
    """List all unchecked Notion to-dos.
    Returns JSON list of {block_id, text}. Call once after creating new to-dos."""
    tasks = NOTION.list_unchecked()
    out = [
        {"block_id": t.get("block_id", t.get("id")), "text": (t.get("text") or "").strip()}
        for t in tasks
    ]
    return json.dumps(out, ensure_ascii=False)
