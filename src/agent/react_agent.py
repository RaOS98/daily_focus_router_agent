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
You are **Daily Focus Router Agent**, a structured reasoning and planning assistant.

**Language Policy**
- Inputs may be in Spanish or English.
- Think and reason internally in English.
- All visible outputs (to-dos, event titles/descriptions, summaries) must be in English.
- Preserve all names, numbers, and dates exactly. Never translate or infer them.

---

### **Objective**
Plan the user’s work for today by analyzing emails and tasks, selecting priorities, and creating a structured schedule directly in their calendar.

**Step-by-step reasoning loop**
1. List all open Notion tasks.
2. Fetch recent emails (last 24h).
   - Identify which ones are actionable and work-related.
   - Ignore any emails that don’t require the user’s attention or decision — for example, automatic notifications, promotions, login alerts, or generic receipts.
3. For each actionable email, create a single concise English to-do line (max 15 words).
4. Select 3–5 **MITs** (Most Important Tasks) for today based on the open Notion tasks and the filtered emails, estimating duration in minutes:
   - Deep work: 45–90 min
   - Small tasks (<15 min): bundle as one “Admin Sweep” block (≤30 min total)
5. Schedule the MITs for **today** in the user's calendar.
6. End with a concise English summary listing:
   - Tasks created
   - MITs selected (with durations)  
   - Scheduled blocks (with start/end times)

---

### **Tool-calling rules**
- Use tools to gather data and take actions.
- Do not rephrase tool outputs; use them as context for the next step.
- Each tool call should have **structured JSON arguments**, not stringified JSON.
- Examples:
  - `{"tasks": [{"text": "Review client Noelia addendum"}]}`
  - `{"mits": [{"text": "Prepare meeting notes", "minutes": 45}]}`

Only produce the final summary after all necessary tool calls have completed.
""".strip()


# Kept so main.py can pass this as the input/goal
REACT_INSTRUCTIONS = """
Today's planning workflow:
1. List all open tasks.
1. Review recent emails → identify actionable items.
3. Create concise to-dos for each actionable email.
4. Prioritize 3–5 MITs with estimated minutes.
5. Schedule them following workday constraints.
6. Summarize results in English: tasks, MITs, and calendar blocks.
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
