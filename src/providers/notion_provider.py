from __future__ import annotations
from typing import Any, Dict, List

from ..config import NOTION_TOKEN, NOTION_TASKS_PAGE_ID

try:
    from notion_client import Client as NotionClient
except Exception:
    NotionClient = None

class NotionProvider:
    def __init__(self):
        self.enabled = NotionClient is not None and NOTION_TOKEN and NOTION_TASKS_PAGE_ID
        self.client = NotionClient(auth=NOTION_TOKEN) if self.enabled else None
        if not self.enabled:
            print("[Notion] MOCK mode (missing NOTION_TOKEN or NOTION_TASKS_PAGE_ID).")
        self.mock_tasks: List[Dict[str, Any]] = []

    def list_unchecked(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            return [t for t in self.mock_tasks if not t.get('checked', False)]
        children = self.client.blocks.children.list(NOTION_TASKS_PAGE_ID)
        tasks = []
        for blk in children.get('results', []):
            if blk['type'] == 'to_do' and not blk['to_do'].get('checked', False):
                text = ''.join([r['plain_text'] for r in blk['to_do']['rich_text']])
                tasks.append({'block_id': blk['id'], 'text': text, 'checked': False})
        return tasks

    def add_todo(self, text: str) -> str:
        if not self.enabled:
            block_id = f"mock_{len(self.mock_tasks)+1}"
            self.mock_tasks.append({'block_id': block_id, 'text': text, 'checked': False})
            print(f"[Notion][MOCK] + To-Do: {text}")
            return block_id
        payload = {'to_do': {'rich_text': [{'text': {'content': text}}], 'checked': False}}
        res = self.client.blocks.children.append(NOTION_TASKS_PAGE_ID, children=[payload])
        return res['results'][0]['id']
