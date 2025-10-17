from __future__ import annotations
import json
from datetime import datetime, timedelta, time
from typing import List, Tuple, Dict, Any

from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from ..config import (
    OLLAMA_MODEL, OLLAMA_BASE_URL, TZ,
    WORKDAY_START, WORKDAY_LUNCH, WORKDAY_END,
    BUFFER_MIN, MAX_BLOCKS, MAX_DEEP_MORNING,
)
from ..providers.calendar_provider import CalendarProvider

CAL = CalendarProvider()


# --- Models -------------------------------------------------------------------

class PrioritizedTask(BaseModel):
    text: str
    minutes: int = Field(..., ge=10, le=120)


# --- Tools --------------------------------------------------------------------

@tool("prioritize_mits", return_direct=False)
def prioritize_mits(tasks: List[Dict[str, Any]]) -> str:
    """
    Choose 3–5 Most Important Tasks for TODAY with estimated durations (minutes).
    Input param `tasks` is a JSON array of objects: [{"text": "..."}, ...]
    Output: JSON string array of {text, minutes}. Call at most once per run.

    Rules:
    - Deep work: 45–90 minutes each.
    - Very small tasks (<15m): bundle into a single "Admin Sweep" block (keep total <=30m).
    - Reason internally in English. Return ONLY valid JSON (no prose).
    """
    # Basic hygiene on inputs
    task_texts = [str(t.get("text", "")).strip() for t in tasks if str(t.get("text", "")).strip()]
    if not task_texts:
        return json.dumps([], ensure_ascii=False)

    # Chat model (messages-based, v1 style)
    model = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )

    system = SystemMessage(content=(
        "You are a productivity assistant. Inputs may be Spanish; reason internally in English. "
        "Pick 3–5 Most Important Tasks for TODAY and estimate minutes per task, following:\n"
        "- Deep work: 45–90 minutes each.\n"
        "- Very small tasks (<15m): bundle into one block called \"Admin Sweep\" (total <=30m).\n"
        "Return ONLY valid JSON array: [{\"text\": str, \"minutes\": int}] with no extra text."
    ))
    human = HumanMessage(content="TASKS:\n" + json.dumps(task_texts, ensure_ascii=False))

    resp = model.invoke([system, human])
    raw = getattr(resp, "content", "") or str(resp)

    # Helper: extract a top-level JSON array if model surrounds it with prose
    def _extract_json_array(s: str) -> str:
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            return s[start:end + 1]
        return "[]"

    raw_json = _extract_json_array(raw)

    # Parse and validate; fall back safely if needed
    try:
        proposed = json.loads(raw_json)
        cleaned: List[Dict[str, Any]] = []
        for item in proposed:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            minutes = int(item.get("minutes", 60))
            # clamp minutes
            minutes = max(10, min(120, minutes))
            obj = PrioritizedTask(text=text, minutes=minutes)
            cleaned.append(obj.model_dump())

        # Cap to 5 items max
        if len(cleaned) > 5:
            cleaned = cleaned[:5]

        return json.dumps(cleaned, ensure_ascii=False)
    except Exception:
        # Minimal deterministic fallback: first up to 3 tasks @ 60m
        fallback = [{"text": t, "minutes": 60} for t in task_texts[:3]]
        return json.dumps(fallback, ensure_ascii=False)


@tool("schedule_blocks", return_direct=False)
def schedule_blocks(mits: List[Dict[str, Any]]) -> str:
    """
    Schedule today's blocks for the given MITs.
    Input param `mits` is a JSON array of objects: [{"text": "...", "minutes": 60}, ...]
    Output: JSON string array of {title, start, end}.
    Call once at the end of the run (after prioritization).

    Constraints:
    - Workday window: 08:30–19:00.
    - Lunch break: 13:00–14:00 (do not schedule across it).
    - Add a 10-minute buffer after each block.
    - <= 5 total blocks today.
    - <= 3 deep-work blocks (>=45m) before 12:00.
    """
    # Input hygiene
    if not mits:
        return json.dumps([], ensure_ascii=False)

    # Normalize & clamp minutes
    normalized: List[Dict[str, Any]] = []
    for m in mits:
        title = str(m.get("text", "")).strip()
        if not title:
            continue
        try:
            dur = int(m.get("minutes", 60))
        except Exception:
            dur = 60
        dur = max(10, min(120, dur))
        normalized.append({"text": title, "minutes": dur})

    if not normalized:
        return json.dumps([], ensure_ascii=False)

    # Time anchors
    now = datetime.now(TZ)
    today = now
    day_start = TZ.localize(datetime.combine(today.date(), WORKDAY_START))
    lunch_start = TZ.localize(datetime.combine(today.date(), WORKDAY_LUNCH[0]))
    lunch_end = TZ.localize(datetime.combine(today.date(), WORKDAY_LUNCH[1]))
    day_end = TZ.localize(datetime.combine(today.date(), WORKDAY_END))

    # Busy times (calendar + lunch)
    busy: List[Tuple[datetime, datetime]] = CAL.get_busy(today) + [(lunch_start, lunch_end)]

    def free_segments(start: datetime, end: datetime, busy_list: List[Tuple[datetime, datetime]]):
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

    created: List[Dict[str, Any]] = []
    blocks_left = MAX_BLOCKS
    deep_morning_left = MAX_DEEP_MORNING

    cursor = max(now, day_start)
    created_busy: List[Tuple[datetime, datetime]] = []

    for mit in normalized:
        if blocks_left <= 0:
            break
        dur = mit["minutes"]
        title = mit["text"]
        is_deep = dur >= 45

        # if morning deep work quota is exhausted, jump cursor to 12:00
        if is_deep and cursor.time() < time(12, 0) and deep_morning_left <= 0:
            cursor = TZ.localize(datetime.combine(today.date(), time(12, 0)))

        placed = False
        cur_busy = busy + created_busy

        for (fs, fe) in free_segments(cursor, day_end, cur_busy):
            # respect lunch: avoid straddling
            segment_minutes = int((fe - fs).total_seconds() // 60)

            if fs < lunch_end and fe > lunch_start:
                # before lunch part
                before_minutes = int((lunch_start - fs).total_seconds() // 60) if fs < lunch_start else 0
                if before_minutes >= (dur + BUFFER_MIN):
                    slot_start, slot_end = fs, fs + timedelta(minutes=dur)
                else:
                    # after lunch part
                    fs2 = max(fs, lunch_end)
                    after_minutes = int((fe - fs2).total_seconds() // 60)
                    if after_minutes < (dur + BUFFER_MIN):
                        continue
                    slot_start, slot_end = fs2, fs2 + timedelta(minutes=dur)
            else:
                if segment_minutes < (dur + BUFFER_MIN):
                    continue
                slot_start, slot_end = fs, fs + timedelta(minutes=dur)

            # Create the event
            desc = (
                "Rules: silent mode, no multitasking.\n"
                "Acceptance: leave minimal evidence (note/link)."
            )
            evt_id = CAL.create_event(title=title, start=slot_start, end=slot_end, description=desc)
            created.append(
                {
                    "calendar_event_id": evt_id,
                    "title": title,
                    "start": slot_start.isoformat(),
                    "end": slot_end.isoformat(),
                }
            )

            # Add the block and a buffer as busy
            buf_start, buf_end = slot_end, slot_end + timedelta(minutes=BUFFER_MIN)
            created_busy.extend([(slot_start, slot_end), (buf_start, buf_end)])

            blocks_left -= 1
            if is_deep and slot_start.time() < time(12, 0):
                deep_morning_left -= 1
            cursor = buf_end
            placed = True
            break  # next MIT

        if not placed:
            # couldn't place this MIT — skip to next
            continue

    # If nothing scheduled at all, drop a short triage to avoid “no plan” days
    if not created:
        triage_dur = 30
        cur_busy = busy + created_busy
        for (fs, fe) in free_segments(cursor, day_end, cur_busy):
            if int((fe - fs).total_seconds() // 60) >= triage_dur:
                evt_id = CAL.create_event(
                    "Triage (30m)",
                    fs,
                    fs + timedelta(minutes=triage_dur),
                    "Sort backlog + minimal plan",
                )
                created.append(
                    {
                        "calendar_event_id": evt_id,
                        "title": "Triage (30m)",
                        "start": fs.isoformat(),
                        "end": (fs + timedelta(minutes=triage_dur)).isoformat(),
                    }
                )
                break

    return json.dumps(created, ensure_ascii=False)
