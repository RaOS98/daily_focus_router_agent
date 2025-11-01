from __future__ import annotations
from typing import Any, Dict, List, TypedDict, Annotated
import operator
import json

from langgraph.graph import StateGraph, START, END

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from ..config import OLLAMA_MODEL, OLLAMA_BASE_URL
from ..tools.email_tools import fetch_recent_emails
from ..tools.notion_tools import add_notion_todo, list_unchecked_tasks
from ..tools.planning_tools import prioritize_mits, schedule_blocks
from ..tools.triage_tools import call_filtering_agent
from ..utils.json_utils import extract_json_array


# State Schema
class WorkflowState(TypedDict, total=False):
    """State for the daily focus router workflow
    
    All fields are optional to allow nodes to initialize state incrementally.
    """
    unchecked_tasks: List[Dict[str, Any]]  # List of unchecked tasks from Notion
    recent_emails: List[Dict[str, Any]]  # List of recent emails from Gmail
    filtered_emails: List[Dict[str, Any]]  # List of filtered actionable emails
    selected_mits: List[Dict[str, Any]]  # List of selected MITs (before time estimation)
    created_todos: Annotated[List[str], operator.add]  # Append new todos
    prioritized_mits: List[Dict[str, Any]]  # List of prioritized MITs with time estimates
    scheduled_blocks: List[Dict[str, Any]]  # List of scheduled calendar blocks
    summary: str  # Final summary text


# Node Functions

def list_tasks_node(state: WorkflowState) -> Dict[str, Any]:
    """Node: List all unchecked tasks from Notion"""
    print("[Node] Listing unchecked tasks...")
    tasks_json = list_unchecked_tasks("")
    tasks = json.loads(tasks_json) if isinstance(tasks_json, str) else tasks_json
    return {"unchecked_tasks": tasks}


def fetch_emails_node(state: WorkflowState) -> Dict[str, Any]:
    """Node: Fetch recent emails from Gmail"""
    print("[Node] Fetching recent emails...")
    emails = fetch_recent_emails()
    return {"recent_emails": emails}


def filter_emails_node(state: WorkflowState) -> Dict[str, Any]:
    """Node: Filter emails using the filtering agent"""
    print("[Node] Filtering emails...")
    emails = state.get("recent_emails", [])
    filtered = call_filtering_agent(emails)
    return {"filtered_emails": filtered}


def select_mits_node(state: WorkflowState) -> Dict[str, Any]:
    """Node: Select MITs from unchecked tasks and filtered emails"""
    print("[Node] Selecting MITs from tasks and emails...")
    
    # Combine unchecked tasks and filtered emails
    all_candidates = []
    
    # Add unchecked tasks
    for task in state.get("unchecked_tasks", []):
        all_candidates.append({
            "text": task.get("text", ""),
            "notion_block_id": task.get("block_id"),
            "source": "notion",
        })
    
    # Add filtered emails
    filtered_emails = state.get("filtered_emails", [])
    for email in filtered_emails:
        thread_id = email.get("thread_id", "")
        subject = email.get("subject", "Untitled")
        snippet = email.get("snippet", "")[:200]
        all_candidates.append({
            "text": subject[:200],
            "thread_id": thread_id,
            "snippet": snippet,
            "source": "email",
        })
    
    if not all_candidates:
        return {"selected_mits": []}
    
    # Use LLM to select 3-5 most important items (without time estimates yet)
    model = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)
    
    system = SystemMessage(content=(
        "You are a productivity assistant. Inputs may be Spanish; reason internally in English. "
        "Select 3–5 Most Important Tasks (MITs) for TODAY from the given candidates. "
        "Focus on urgency, importance, and alignment with daily goals.\n"
        "Return ONLY a valid JSON array of the selected task texts: [{\"text\": str}] with no extra text."
    ))
    
    candidates_text = json.dumps(
        [{"text": c["text"], "source": c["source"]} for c in all_candidates],
        ensure_ascii=False
    )
    human = HumanMessage(content=f"CANDIDATES:\n{candidates_text}")
    
    resp = model.invoke([system, human])
    raw = getattr(resp, "content", "") or str(resp)
    
    arr = extract_json_array(raw)
    
    # Match selected texts back to candidates to preserve metadata
    selected_mits = []
    selected_texts = {item.get("text", "") for item in arr if isinstance(item, dict)}
    
    for candidate in all_candidates:
        if candidate["text"] in selected_texts:
            # Preserve all metadata
            selected_mit = {
                "text": candidate["text"],
                "source": candidate["source"],
            }
            if candidate.get("thread_id"):
                selected_mit["thread_id"] = candidate["thread_id"]
            if candidate.get("notion_block_id"):
                selected_mit["notion_block_id"] = candidate["notion_block_id"]
            if candidate.get("snippet"):
                selected_mit["snippet"] = candidate["snippet"]
            selected_mits.append(selected_mit)
    
    # Cap to 5 items max
    if len(selected_mits) > 5:
        selected_mits = selected_mits[:5]
    
    return {"selected_mits": selected_mits}


def add_todos_node(state: WorkflowState) -> Dict[str, Any]:
    """Node: Add tasks to Notion only for email-based selected MITs"""
    print("[Node] Adding todos for selected email MITs...")
    selected_mits = state.get("selected_mits", [])
    filtered_emails = state.get("filtered_emails", [])
    created_todos = []
    
    # Create a lookup map of emails by thread_id
    emails_by_thread = {email.get("thread_id"): email for email in filtered_emails}
    
    # Only add todos for email-based selected MITs
    for mit in selected_mits:
        if mit.get("source") != "email":
            continue
        
        thread_id = mit.get("thread_id")
        if not thread_id:
            continue
        
        email = emails_by_thread.get(thread_id)
        if not email:
            continue
        
        subject = email.get("subject", "Untitled")
        snippet = email.get("snippet", "")[:200]
        
        # Create concise title (<= 15 words)
        title_parts = subject.split()[:15]
        if snippet:
            snippet_parts = snippet.split()[:10]
            title = " ".join(title_parts + snippet_parts[:15-len(title_parts)])
        else:
            title = " ".join(title_parts)
        
        # Create payload with thread_id for state linking
        payload = json.dumps({
            "text": title[:200],  # Limit length
            "thread_id": thread_id
        })
        
        try:
            block_id = add_notion_todo(payload)
            created_todos.append(block_id)
        except Exception as e:
            print(f"[Warning] Failed to add todo for {thread_id}: {e}")
            continue
    
    return {"created_todos": created_todos}


def prioritize_mits_node(state: WorkflowState) -> Dict[str, Any]:
    """Node: Add time estimates to selected MITs"""
    print("[Node] Adding time estimates to selected MITs...")
    
    selected_mits = state.get("selected_mits", [])
    if not selected_mits:
        return {"prioritized_mits": []}
    
    # Convert selected_mits to format expected by prioritize_mits tool
    tasks_for_estimation = []
    for mit in selected_mits:
        task_dict = {"text": mit.get("text", "")}
        if mit.get("thread_id"):
            task_dict["thread_id"] = mit["thread_id"]
        if mit.get("notion_block_id"):
            task_dict["notion_block_id"] = mit["notion_block_id"]
        tasks_for_estimation.append(task_dict)
    
    # Call prioritize_mits tool to add time estimates
    mits_json = prioritize_mits(tasks_for_estimation)
    mits = json.loads(mits_json) if isinstance(mits_json, str) else mits_json
    
    return {"prioritized_mits": mits}


def schedule_blocks_node(state: WorkflowState) -> Dict[str, Any]:
    """Node: Schedule calendar blocks for MITs"""
    print("[Node] Scheduling calendar blocks...")
    
    mits = state.get("prioritized_mits", [])
    if not mits:
        return {"scheduled_blocks": []}
    
    # Call schedule_blocks tool
    blocks_json = schedule_blocks(mits)
    blocks = json.loads(blocks_json) if isinstance(blocks_json, str) else blocks_json
    
    return {"scheduled_blocks": blocks}


def generate_summary_node(state: WorkflowState) -> Dict[str, Any]:
    """Node: Generate final summary"""
    print("[Node] Generating summary...")
    
    unchecked_count = len(state.get("unchecked_tasks", []))
    emails_count = len(state.get("recent_emails", []))
    filtered_count = len(state.get("filtered_emails", []))
    selected_count = len(state.get("selected_mits", []))
    todos_count = len(state.get("created_todos", []))
    mits = state.get("prioritized_mits", [])
    blocks = state.get("scheduled_blocks", [])
    
    summary_parts = [
        "===== DAILY FOCUS PLAN SUMMARY =====",
        f"\n📋 Found {unchecked_count} unchecked tasks from Notion",
        f"\n📧 Fetched {emails_count} emails (last 24h), filtered to {filtered_count} actionable",
        f"\n🎯 Selected {selected_count} Most Important Tasks (MITs) from combined sources",
        f"\n✅ Created {todos_count} new to-dos in Notion for selected email MITs",
        f"\n⏱️  Prioritized {len(mits)} MITs with time estimates:",
    ]
    
    for i, mit in enumerate(mits, 1):
        minutes = mit.get("minutes", 0)
        text = mit.get("text", "")
        summary_parts.append(f"   {i}. {text} ({minutes} min)")
    
    summary_parts.append(f"\n📅 Scheduled {len(blocks)} calendar blocks:")
    for block in blocks:
        title = block.get("title", "")
        start = block.get("start", "")
        end = block.get("end", "")
        summary_parts.append(f"   • {title}: {start} - {end}")
    
    summary = "\n".join(summary_parts)
    
    return {"summary": summary}


def build_graph():
    """Build and compile the LangGraph workflow with parallel execution"""
    # Create the graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("list_tasks", list_tasks_node)
    workflow.add_node("fetch_emails", fetch_emails_node)
    workflow.add_node("filter_emails", filter_emails_node)
    workflow.add_node("select_mits", select_mits_node)
    workflow.add_node("add_todos", add_todos_node)
    workflow.add_node("prioritize", prioritize_mits_node)
    workflow.add_node("schedule", schedule_blocks_node)
    workflow.add_node("summary", generate_summary_node)
    
    # Parallel execution: fan-out from START
    # Branch 1: list_tasks (runs in parallel)
    workflow.add_edge(START, "list_tasks")
    
    # Branch 2: fetch_emails → filter_emails (runs in parallel with branch 1)
    workflow.add_edge(START, "fetch_emails")
    workflow.add_edge("fetch_emails", "filter_emails")
    
    # Fan-in: both branches complete, then select_mits waits for both
    workflow.add_edge("list_tasks", "select_mits")
    workflow.add_edge("filter_emails", "select_mits")
    
    # Sequential flow after selection
    workflow.add_edge("select_mits", "add_todos")
    workflow.add_edge("add_todos", "prioritize")
    workflow.add_edge("prioritize", "schedule")
    workflow.add_edge("schedule", "summary")
    workflow.add_edge("summary", END)
    
    # Compile the graph
    return workflow.compile()

