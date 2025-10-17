from __future__ import annotations
import json
from langchain.tools import tool

from ..providers.notion_provider import NotionProvider

NOTION = NotionProvider()

@tool("add_notion_todo", return_direct=False)
def add_notion_todo(task_text: str) -> str:
    """Create a checkbox to-do in Notion 'Tasks' page. Input: a concise one-line task text (Spanish ok). Returns the Notion block_id."""
    block_id = NOTION.add_todo(task_text)
    return block_id

@tool("list_unchecked_tasks", return_direct=False)
def list_unchecked_tasks(_: str = "") -> str:
    """List all unchecked Notion to-dos. Input ignored. Returns JSON list of {block_id, text}."""
    tasks = NOTION.list_unchecked()
    return json.dumps([{"block_id": t.get('block_id', t.get('id')), "text": t['text']} for t in tasks], ensure_ascii=False)
