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
    call_filtering_agent,
)


SYSTEM_PROMPT = """
You are **Daily Focus Router Agent**, a structured reasoning and planning assistant.

## Language
- Inputs may be Spanish or English.
- Think internally in English.
- All user-visible outputs (to-dos, calendar titles/descriptions, summaries) must be English.
- Preserve all names, numbers, and dates exactly (never translate or modify them).

## Mission
Plan the user’s work for today by analyzing emails and tasks, creating concise to-dos, selecting 3–5 Most Important Tasks (MITs), and scheduling them in the calendar.
""".strip()


REACT_INSTRUCTIONS = """
Follow this exact sequence.

1) List unchecked tasks.

2) Fetch recent emails.

3) Call filtering agent.

4) Add tasks from filtered emails to to-do list with a concise title (<= 15 words).

5) Prioritize MITs from the combined list:
   - All open Notion tasks from step 1.
   - New to-dos you created in step 4.

6) Schedule blocks in user's calendar for the MITs returned by step 5, allocating appropriate time based on estimated effort.

7) Write the final English summary (tasks created, MITs with minutes, scheduled blocks with start/end). Then stop.
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
        call_filtering_agent,
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
