from __future__ import annotations
import json
from datetime import datetime, timedelta, time
from typing import List, Tuple

from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from ..config import (
    OLLAMA_MODEL, OLLAMA_BASE_URL, TZ,
    WORKDAY_START, WORKDAY_LUNCH, WORKDAY_END,
    BUFFER_MIN, MAX_BLOCKS, MAX_DEEP_MORNING,
)
from ..providers.calendar_provider import CalendarProvider

CAL = CalendarProvider()

class PrioritizedTask(BaseModel):
    text: str
    minutes: int = Field(..., ge=10, le=120)

@tool("prioritize_mits", return_direct=False)
def prioritize_mits(tasks_json: str) -> str:
    """Given JSON list of tasks [{'text': ...}], pick 3–5 Most Important Tasks for TODAY with an estimated duration in minutes (45–90 if deep work, <=15 bundle as 'Admin Sweep'). Return JSON list of {text, minutes}."""
    tasks = json.loads(tasks_json)
    llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    prompt = PromptTemplate.from_template(
        """
        Eres un asistente de productividad. Recibes una lista de tareas (solo texto). Objetivo: elige 3–5 MITs para HOY y estima su duración:
        - Tareas profundas: 45–90 min cada una.
        - Tareas pequeñas (<15m): júntalas en un solo bloque llamado "Admin Sweep" con la suma total (<=30m si posible).
        Devuelve SOLO JSON válido: [{{"text": str, "minutes": int}}, ...]

        TAREAS:
        {tasks}
        """
    )
    raw = llm.invoke(prompt.format(tasks=json.dumps([t['text'] for t in tasks], ensure_ascii=False)))
    try:
        data = json.loads(raw)
        cleaned = []
        for item in data:
            obj = PrioritizedTask(text=str(item.get('text')).strip(), minutes=int(item.get('minutes', 60)))
            cleaned.append(obj.model_dump())
        return json.dumps(cleaned, ensure_ascii=False)
    except Exception:
        fallback = [{"text": tasks[i]['text'], "minutes": 60} for i in range(min(3, len(tasks)))]
        return json.dumps(fallback, ensure_ascii=False)

@tool("schedule_blocks", return_direct=False)
def schedule_blocks(mits_json: str) -> str:
    """Schedule the given MITs TODAY respecting workday (08:30–19:00, lunch 13:00–14:00), 10m buffers, <=5 blocks/day, <=3 deep-work in morning. Input: JSON list of {text, minutes}. Returns JSON of created events with {title,start,end}."""
    mits = json.loads(mits_json)
    today = datetime.now(TZ)
    day_start = TZ.localize(datetime.combine(today.date(), WORKDAY_START))
    lunch_start = TZ.localize(datetime.combine(today.date(), WORKDAY_LUNCH[0]))
    lunch_end = TZ.localize(datetime.combine(today.date(), WORKDAY_LUNCH[1]))
    day_end = TZ.localize(datetime.combine(today.date(), WORKDAY_END))

    busy = CAL.get_busy(today) + [(lunch_start, lunch_end)]

    def free_segments(start, end, busy_list: List[Tuple[datetime, datetime]]):
        bsorted = sorted(busy_list, key=lambda x: x[0])
        cur = start
        for b in bsorted:
            if b[0] > cur:
                yield (cur, min(b[0], end))
            cur = max(cur, b[1])
            if cur >= end:
                break
        if cur < end:
            yield (cur, end)

    created = []
    blocks_left = MAX_BLOCKS
    deep_morning_left = MAX_DEEP_MORNING

    now = datetime.now(TZ)
    cursor = max(now, day_start)
    created_busy: List[Tuple[datetime, datetime]] = []

    for mit in mits:
        if blocks_left <= 0:
            break
        dur = int(mit['minutes'])
        title = mit['text']
        is_deep = dur >= 45
        if is_deep and cursor.time() < time(12, 0) and deep_morning_left <= 0:
            cursor = TZ.localize(datetime.combine(today.date(), time(12, 0)))
        placed = False
        cur_busy = busy + created_busy
        for (fs, fe) in free_segments(cursor, day_end, cur_busy):
            if fs < lunch_end and fe > lunch_start:
                if fs < lunch_start and (lunch_start - fs).total_seconds() / 60 >= dur + BUFFER_MIN:
                    slot_start, slot_end = fs, fs + timedelta(minutes=dur)
                else:
                    fs = max(fs, lunch_end)
                    if (fe - fs).total_seconds() / 60 < dur + BUFFER_MIN:
                        continue
                    slot_start, slot_end = fs, fs + timedelta(minutes=dur)
            else:
                if (fe - fs).total_seconds() / 60 < dur + BUFFER_MIN:
                    continue
                slot_start, slot_end = fs, fs + timedelta(minutes=dur)

            desc = (
                "Reglas: sin notifs, sin multitarea.\n"
                "Criterio de aceptación: dejar evidencia mínima (nota/enlace)."
            )
            evt_id = CAL.create_event(title=title, start=slot_start, end=slot_end, description=desc)
            created.append({'calendar_event_id': evt_id, 'title': title, 'start': slot_start.isoformat(), 'end': slot_end.isoformat()})
            buf_start, buf_end = slot_end, slot_end + timedelta(minutes=BUFFER_MIN)
            created_busy.extend([(slot_start, slot_end), (buf_start, buf_end)])

            blocks_left -= 1
            if is_deep and slot_start.time() < time(12, 0):
                deep_morning_left -= 1
            cursor = slot_end + timedelta(minutes=BUFFER_MIN)
            placed = True
            break
        if not placed:
            continue

    if not created:
        triage_dur = 30
        for (fs, fe) in free_segments(cursor, day_end, busy + created_busy):
            if (fe - fs).total_seconds() / 60 >= triage_dur:
                evt_id = CAL.create_event("Triage (30m)", fs, fs + timedelta(minutes=triage_dur), "Clasificar pendientes + plan mínimo")
                created.append({'calendar_event_id': evt_id, 'title': 'Triage (30m)', 'start': fs.isoformat(), 'end': (fs+timedelta(minutes=triage_dur)).isoformat()})
                break

    return json.dumps(created, ensure_ascii=False)
