from __future__ import annotations
from typing import Any, Dict

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from ..config import OLLAMA_MODEL, OLLAMA_BASE_URL
from ..tools import (
    fetch_recent_emails,
    add_notion_todo,
    list_unchecked_tasks,
    prioritize_mits,
    schedule_blocks,
)

SYSTEM_PROMPT = """
You are "Daily Focus Router Agent".

Language policy:
- Inputs may be Spanish or English. Internally reason in English.
- All user-visible outputs (to-dos, calendar titles/descriptions, final summary) must be English.
- Preserve names, numbers, and dates exactly (do not translate those). Quote short spans if unsure.

Goal for TODAY:
1) Fetch recent emails and decide which are ACTIONABLE (review, approve, send, decide, coordinate; today/tomorrow/this week).
   Ignore newsletters and invoices with no explicit ask.
2) For each actionable email, create ONE concise to-do line in English.
3) List all open tasks.
4) Choose 3–5 MITs for today with estimated minutes (bundle <15m items into one "Admin Sweep" ≤30m total).
5) Schedule blocks today respecting: 08:30–19:00, lunch 13:00–14:00, 10' buffers, ≤5 blocks/day, ≤3 deep-work in the morning.
6) End with a clear English summary: tasks created, MITs chosen, scheduled blocks (start/end).

Use available tools to gather facts and take actions. Finish with a final answer only after necessary tool calls.

Stop condition (VERY IMPORTANT):
- After you call `schedule_blocks`, DO NOT call any more tools. Immediately produce your final natural-language answer.
- If there are zero actionable emails and zero tasks to schedule, DO NOT call tools repeatedly. Immediately produce your final natural-language answer stating there is nothing to schedule.
- Never call the same tool twice for the same purpose (e.g., do not repeatedly call fetch_recent_emails or list_unchecked_tasks).

""".strip()

# Kept so main.py can pass this as the input/goal
REACT_INSTRUCTIONS = """
Plan for TODAY:
- Triage recent emails → actionable items only.
- Create concise to-dos in English for actionable emails.
- List open tasks, select 3–5 MITs with minutes.
- Schedule blocks per day rules.
- Summarize outcomes (tasks, MITs, schedule) in English.
""".strip()


class _SimpleAgentAdapter:
    def __init__(self, runnable):
        self._runnable = runnable

    def invoke(self, inputs):
        from langchain_core.messages import HumanMessage, AIMessage
        user_text = inputs.get("input", "")

        res = self._runnable.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config={"recursion_limit": 50},
        )

        # Normalize to the old {"output": "..."} shape
        content = ""
        msgs = res.get("messages") if isinstance(res, dict) else None
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            # AIMessage or dict-like
            if isinstance(last, AIMessage):
                content = last.content or ""
            else:
                content = (getattr(last, "content", None)
                           or (isinstance(last, dict) and last.get("content"))
                           or "")
        elif isinstance(res, str):
            content = res
        else:
            # fallback: try common fields
            content = res.get("output", "") if isinstance(res, dict) else ""

        return {"output": content}


def build_executor():
    tools = [
        fetch_recent_emails,
        add_notion_todo,
        list_unchecked_tasks,
        prioritize_mits,
        schedule_blocks,
    ]

    # Chat model (tool-calling capable). Set OLLAMA_MODEL=gpt-oss:20b or similar.
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )

    # Modern tool-calling agent (no fragile Thought/Action parsing)
    agent_runnable = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    # Return adapter with .invoke(...) so main.py keeps working
    return _SimpleAgentAdapter(agent_runnable)
