from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from ..config import TZ, GOOGLE_CREDS_PATH, GOOGLE_SCOPES

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
except Exception:
    Credentials = None
    InstalledAppFlow = None
    build = None

@dataclass
class Email:
    thread_id: str
    subject: str
    snippet: str
    sender: str
    date: str

    def short(self) -> str:
        return f"[{self.thread_id}] {self.subject} — {self.snippet[:120]}"

class GmailProvider:
    def __init__(self):
        self.enabled = Credentials is not None and GOOGLE_CREDS_PATH.exists()
        self.service = self._ensure_service() if self.enabled else None
        if not self.enabled:
            print("[Gmail] MOCK mode (no credentials found).")

    def _ensure_service(self):
        if not self.enabled:
            return None
        creds = None
        token_path = GOOGLE_CREDS_PATH.parent / "token.json"
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), GOOGLE_SCOPES)
        if not creds or not creds.valid:
            if InstalledAppFlow is None:
                return None
            flow = InstalledAppFlow.from_client_secrets_file(str(GOOGLE_CREDS_PATH), GOOGLE_SCOPES)
            creds = flow.run_local_server(port=0)
            token_path.write_text(creds.to_json())
        return build('gmail', 'v1', credentials=creds)

    def fetch_last_24h(self, max_results: int = 30) -> List[Dict[str, Any]]:
        if not self.enabled or self.service is None:
            now = datetime.now(TZ)
            return [
                {"thread_id": "t1", "subject": "Adenda Minuta Cliente Noelia", "snippet": "Por favor revisar y aprobar la adenda.", "from": "abogada@example.com", "date": now.isoformat()},
                {"thread_id": "t2", "subject": "Factura de Compras", "snippet": "Adjuntamos PDF.", "from": "facturas@example.com", "date": now.isoformat()},
                {"thread_id": "t3", "subject": "Coordinar visita terreno", "snippet": "¿Podemos coordinar para este viernes?", "from": "lead@example.com", "date": now.isoformat()},
            ]
        svc = self.service
        user_id = 'me'
        q = 'newer_than:2d'
        res = svc.users().messages().list(userId=user_id, q=q, maxResults=max_results).execute()
        ids = [m['id'] for m in res.get('messages', [])]
        out = []
        for msg_id in ids:
            msg = svc.users().messages().get(userId=user_id, id=msg_id, format='metadata', metadataHeaders=['Subject', 'From', 'Date']).execute()
            headers = {h['name']: h['value'] for h in msg['payload'].get('headers', [])}
            subject = headers.get('Subject', '(sin asunto)')
            sender = headers.get('From', '')
            date = headers.get('Date', '')
            thread_id = msg.get('threadId')
            snippet = msg.get('snippet', '')
            out.append({"thread_id": thread_id, "subject": subject, "snippet": snippet, "from": sender, "date": date})
        return out
