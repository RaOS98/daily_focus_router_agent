from __future__ import annotations
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from ..config import OLLAMA_MODEL, OLLAMA_BASE_URL
from ..tools import (
    fetch_recent_emails,
    add_notion_todo,
    list_unchecked_tasks,
    prioritize_mits,
    schedule_blocks,
)

REACT_PROMPT = PromptTemplate.from_template(
    """
You are "Daily Focus Router Agent". Turn emails + to-dos into a plan for TODAY using ONLY the tools.

Language policy: Inputs may be Spanish or English. Reason internally in English. All outputs (to-dos, calendar titles, final summary) must be English. Preserve names/numbers/dates as-is.

Formatting rules (MANDATORY):
- Respond in steps using this EXACT cycle:
  Thought:
  Action:
  Action Input:
- DO NOT write "Observation" yourself. The framework will run the tool and append "Observation:" automatically.
- NOTHING between Thought and Action (no extra text).
- For tools with no input, use empty string "" in Action Input.
- **Never write 'Final Answer' until AFTER you have called `schedule_blocks` (or you have determined there are zero actionable emails and zero tasks to schedule).**

Tools available:
{tools}
Tool names: {tool_names}

Minimal example (fictional):
Thought: I need to fetch recent emails.
Action: fetch_recent_emails
Action Input: ""

Thought: Create a to-do for the client's addendum.
Action: add_notion_todo
Action Input: "Review and approve client Noelia addendum (today)"

Thought: List open tasks.
Action: list_unchecked_tasks
Action Input: ""

Thought: Prioritize 3–5 MITs with durations.
Action: prioritize_mits
Action Input: "[{{\\"text\\": \\"Review client Noelia addendum\\"}}]"

Thought: Schedule blocks for today.
Action: schedule_blocks
Action Input: "[{{\\"text\\": \\"Review client Noelia addendum\\", \\"minutes\\": 60}}]"

When actions are complete:
Final Answer: (clear English summary: tasks created, MITs, blocks with start/end times)

Your goal/input:
{input}

{agent_scratchpad}
"""
)

REACT_INSTRUCTIONS = """
Follow this flow to plan for TODAY (English-only outputs):
1) fetch_recent_emails → decide which emails are ACTIONABLE (review, approve, send, decide, coordinate; today/tomorrow/this week; addendum/minutes/contract/draft; payments/deadlines). Ignore newsletters and invoices with no ask.
2) For each actionable email, create ONE concise to-do line in English with add_notion_todo.
3) list_unchecked_tasks to get all open tasks.
4) prioritize_mits with those tasks → choose 3–5 MITs and estimate minutes.
5) schedule_blocks with that JSON → respect 08:30–19:00, lunch 13:00–14:00, 10' buffers, ≤5 blocks/day, ≤3 deep work in the morning.
End with a clear English summary: tasks created, MITs chosen, and scheduled blocks (start/end time).
"""

def build_executor() -> AgentExecutor:
    tools = [
        fetch_recent_emails,
        add_notion_todo,
        list_unchecked_tasks,
        prioritize_mits,
        schedule_blocks,
    ]
    # Lower temp + stop sequence helps the model stick to the format:
    llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    react = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)
    executor = AgentExecutor(
        agent=react,
        tools=tools,
        verbose=True,
        handle_parsing_errors=(
        "INVALID FORMAT. Reply using EXACTLY this cycle and DO NOT write 'Observation' "
        "or 'Final Answer' yet:\nThought:\nAction:\nAction Input:\n"
        "Only write 'Final Answer' AFTER calling schedule_blocks (or after deciding there is nothing to schedule)."
        ),
        max_iterations=20,
    )
    return executor
