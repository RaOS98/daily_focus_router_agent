from __future__ import annotations
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

from .config import DB_PATH


class TinyStore:
    def __init__(self, path: Path = DB_PATH):
        # Ensure parent dir exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Allow use across threads; we’ll add our own lock.
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        # Faster, safe enough for this tiny store
        self.conn.execute("PRAGMA synchronous=NORMAL;")

        self._lock = threading.Lock()
        self._init()

    def _init(self):
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS mappings (
                  thread_id TEXT PRIMARY KEY,
                  notion_block_id TEXT,
                  calendar_event_id TEXT,
                  created_at TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS cursors (
                  provider TEXT PRIMARY KEY,
                  cursor TEXT,
                  updated_at TEXT
                )
                """
            )
            self.conn.commit()

    def upsert_mapping(
        self,
        thread_id: str,
        notion_block_id: Optional[str] = None,
        calendar_event_id: Optional[str] = None,
    ):
        ts = datetime.utcnow().isoformat()
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT thread_id, notion_block_id, calendar_event_id FROM mappings WHERE thread_id=?",
                (thread_id,),
            )
            row = cur.fetchone()
            if row:
                nb = notion_block_id or row[1]
                ce = calendar_event_id or row[2]
                cur.execute(
                    "UPDATE mappings SET notion_block_id=?, calendar_event_id=?, created_at=? WHERE thread_id=?",
                    (nb, ce, ts, thread_id),
                )
            else:
                cur.execute(
                    "INSERT INTO mappings(thread_id, notion_block_id, calendar_event_id, created_at) VALUES (?,?,?,?)",
                    (thread_id, notion_block_id, calendar_event_id, ts),
                )
            self.conn.commit()

    def set_calendar_event(self, thread_id: str, calendar_event_id: str):
        """Convenience: attach a calendar event ID to an existing mapping (or create if missing)."""
        self.upsert_mapping(thread_id=thread_id, calendar_event_id=calendar_event_id)

    def get_mapping(self, thread_id: str) -> Dict[str, Optional[str]]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT notion_block_id, calendar_event_id FROM mappings WHERE thread_id=?",
                (thread_id,),
            )
            row = cur.fetchone()
        if not row:
            return {"notion_block_id": None, "calendar_event_id": None}
        return {"notion_block_id": row[0], "calendar_event_id": row[1]}

    def get_cursor(self, provider: str) -> Optional[str]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT cursor FROM cursors WHERE provider=?", (provider,))
            row = cur.fetchone()
        return row[0] if row else None

    def set_cursor(self, provider: str, cursor: str):
        ts = datetime.utcnow().isoformat()
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "REPLACE INTO cursors(provider, cursor, updated_at) VALUES (?,?,?)",
                (provider, cursor, ts),
            )
            self.conn.commit()

    def close(self):
        with self._lock:
            try:
                self.conn.close()
            except Exception:
                pass


STORE = TinyStore()
