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
from ..store import STORE  # <-- state wiring

CAL = CalendarProvider()


# --- Models -------------------------------------------------------------------

class PrioritizedTask(BaseModel):
    text: str
    minutes: int = Field(..., ge=10, le=120)


# Args schemas (schema-first tool calling) -------------------------------------

class PrioritizeArgs(BaseModel):
    # Accept an array of {"text": "..."} items; extra keys allowed (e.g., thread_id)
    tasks: List[Dict[str, Any]] = Field(
        description="List of task objects. Each item must at least have a 'text' field."
    )

class ScheduleArgs(BaseModel):
    # Accept an array of {"text": "...", "minutes": int} items; extra keys allowed
    mits: List[Dict[str, Any]] = Field(
        description="List of MIT objects with 'text' and 'minutes'. Extra fields allowed."
    )


# --- Helper: extract JSON array from LLM text (best-effort) -------------------

def _extract_json_array(s: str) -> List[Any]:
    """Best-effort: pull a top-level JSON array from an arbitrary string."""
    if not isinstance(s, str):
        return []
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except Exception:
            return []
    return []


# --- Tools --------------------------------------------------------------------

@tool("prioritize_mits", args_schema=PrioritizeArgs, return_direct=False)
def prioritize_mits(tasks: List[Dict[str, Any]]) -> str:
    """
    Select 3–5 Most Important Tasks (MITs) for today and estimate durations.

    Args:
        tasks: list[dict] | dict | str
            Prefer a list of {"text": str, ...}. Extra keys (e.g., "thread_id",
            "notion_block_id") may be present and are passed through when the
            returned item's text matches exactly. A dict form {"tasks": [...]} or
            a JSON string of either shape is also accepted.

    Returns:
        str: JSON array string like:
            '[{"text": str, "minutes": int, ...}]'
            Minutes are clamped to 10–120. No prose.

    Notes:
        Bundle sub-15m items into one "Admin Sweep" block (≤30m total).
    """

    print("[prioritize_mits] invoked")

    # Basic hygiene on inputs
    task_texts: List[str] = []
    id_by_text: Dict[str, Dict[str, Any]] = {}
    for it in tasks:
        txt = str(it.get("text", "")).strip()
        if not txt:
            continue
        task_texts.append(txt)
        # keep passthrough ids by exact text key
        id_by_text[txt] = {
            "thread_id": it.get("thread_id"),
            "notion_block_id": it.get("notion_block_id"),
        }

    if not task_texts:
        return json.dumps([], ensure_ascii=False)

    # Chat model
    model = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)

    system = SystemMessage(content=(
        "You are a productivity assistant. Inputs may be Spanish; reason internally in English. "
        "Pick 3–5 Most Important Tasks for TODAY and estimate minutes per task, following:\n"
        "- Deep work: 45–90 minutes each.\n"
        "- Very small tasks (<15m): bundle into one block called \"Admin Sweep\" (total <=30m).\n"
        "Return ONLY a valid JSON array: [{\"text\": str, \"minutes\": int}] with no extra text."
    ))
    human = HumanMessage(content="TASKS:\n" + json.dumps(task_texts, ensure_ascii=False))

    resp = model.invoke([system, human])
    raw = getattr(resp, "content", "") or str(resp)

    arr = _extract_json_array(raw)
    try:
        cleaned: List[Dict[str, Any]] = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            minutes = int(item.get("minutes", 60))
            minutes = max(10, min(120, minutes))

            obj = PrioritizedTask(text=text, minutes=minutes).model_dump()

            # Re-attach ids on exact text match (best-effort)
            ids = id_by_text.get(text)
            if ids:
                if ids.get("thread_id"):
                    obj["thread_id"] = ids["thread_id"]
                if ids.get("notion_block_id"):
                    obj["notion_block_id"] = ids["notion_block_id"]

            cleaned.append(obj)

        # Cap to 5 items max
        if len(cleaned) > 5:
            cleaned = cleaned[:5]

        return json.dumps(cleaned, ensure_ascii=False)
    except Exception:
        # Minimal deterministic fallback: first up to 3 tasks @ 60m
        fallback = [{"text": t, "minutes": 60} for t in task_texts[:3]]
        return json.dumps(fallback, ensure_ascii=False)


@tool("schedule_blocks", args_schema=ScheduleArgs, return_direct=False)
def schedule_blocks(mits: List[Dict[str, Any]]) -> str:
    """
    Create calendar events for today's MITs.

    Args:
        mits: list[dict] | dict | str
            Prefer a list of {"text": str, "minutes": int, ...}. Extra ids
            (e.g., "thread_id", "notion_block_id") may be included and are used
            for state linking. A dict form {"mits": [...]} or a JSON string of
            either shape is also accepted.

    Returns:
        str: JSON array string like:
            '[{"calendar_event_id": str, "title": str, "start": ISO8601, "end": ISO8601}]'

    Notes:
        Work hours, lunch, buffer, and deep-work caps are enforced in code; do not
        restate them in the output.
    """

    print("[schedule_blocks] invoked")

    # Normalize & clamp minutes; preserve passthrough IDs
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
        normalized.append({
            "text": title,
            "minutes": dur,
            "thread_id": m.get("thread_id"),
            "notion_block_id": m.get("notion_block_id"),
        })

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
            record = {
                "calendar_event_id": evt_id,
                "title": title,
                "start": slot_start.isoformat(),
                "end": slot_end.isoformat(),
            }
            created.append(record)

            # STATE: if this MIT came from an email thread, link it to the new calendar event
            if mit.get("thread_id"):
                try:
                    STORE.upsert_mapping(
                        thread_id=str(mit["thread_id"]),
                        calendar_event_id=str(evt_id),
                    )
                except Exception:
                    # don't break scheduling on store failures
                    pass

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
