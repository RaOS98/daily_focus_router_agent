from __future__ import annotations
from datetime import datetime
from typing import List, Tuple

from ..config import TZ, GOOGLE_CREDS_PATH, GOOGLE_SCOPES

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
except Exception:
    Credentials = None
    InstalledAppFlow = None
    build = None

class CalendarProvider:
    def __init__(self):
        self.enabled = Credentials is not None and GOOGLE_CREDS_PATH.exists()
        self.service = self._ensure_service() if self.enabled else None
        if not self.enabled:
            print("[Calendar] MOCK mode (no credentials found).")

    def _ensure_service(self):
        if not self.enabled:
            return None
        token_path = GOOGLE_CREDS_PATH.parent / "token.json"
        creds = None
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), GOOGLE_SCOPES)
        if not creds or not creds.valid:
            if InstalledAppFlow is None:
                return None
            flow = InstalledAppFlow.from_client_secrets_file(str(GOOGLE_CREDS_PATH), GOOGLE_SCOPES)
            creds = flow.run_local_server(port=0)
            token_path.write_text(creds.to_json())
        return build('calendar', 'v3', credentials=creds)

    def get_busy(self, day: datetime) -> List[Tuple[datetime, datetime]]:
        if not self.enabled or self.service is None:
            return []
        start = TZ.localize(datetime.combine(day.date(), datetime.min.time())).isoformat()
        end = TZ.localize(datetime.combine(day.date(), datetime.max.time())).isoformat()
        events = self.service.events().list(calendarId='primary', timeMin=start, timeMax=end, singleEvents=True, orderBy='startTime').execute()
        busy = []
        for e in events.get('items', []):
            s = e['start'].get('dateTime')
            e_ = e['end'].get('dateTime')
            if s and e_:
                ds = datetime.fromisoformat(s)
                de = datetime.fromisoformat(e_)
                busy.append((ds.astimezone(TZ), de.astimezone(TZ)))
        return busy

    def create_event(self, title: str, start: datetime, end: datetime, description: str) -> str:
        if not self.enabled or self.service is None:
            print(f"[Calendar][MOCK] + Event: {start.strftime('%H:%M')}–{end.strftime('%H:%M')} | {title}\n{description}\n")
            return f"mock_evt_{int(start.timestamp())}"
        event = {
            'summary': title,
            'description': description,
            'start': {'dateTime': start.isoformat()},
            'end': {'dateTime': end.isoformat()},
        }
        created = self.service.events().insert(calendarId='primary', body=event).execute()
        return created['id']
