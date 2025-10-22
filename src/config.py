from __future__ import annotations

from dotenv import load_dotenv
load_dotenv() 

import os
from datetime import time
from pathlib import Path
import pytz

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SECRETS_DIR = BASE_DIR / "secrets"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SECRETS_DIR.mkdir(parents=True, exist_ok=True)

# Env
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_TASKS_PAGE_ID = os.getenv("NOTION_TASKS_PAGE_ID")
APP_TZ = os.getenv("APP_TZ", "America/Lima")
TZ = pytz.timezone(APP_TZ)

# Google OAuth
GOOGLE_TOKEN_PATH = SECRETS_DIR / "token.json"
GOOGLE_CREDS_PATH = SECRETS_DIR / "credentials.json"
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar",
]

# Workday rules
WORKDAY_START = time(8, 0)
WORKDAY_LUNCH = (time(13, 0), time(14, 0))
WORKDAY_END = time(19, 0)
BLOCK_MIN = 45
BLOCK_MAX = 90
BUFFER_MIN = 10
MAX_BLOCKS = 5
MAX_DEEP_MORNING = 3

# DB
DB_PATH = DATA_DIR / "tiny_store.sqlite"
