from __future__ import annotations
import sys
from .agent import build_executor

def plan_now() -> str:
    graph = build_executor()
    # LangGraph workflow starts with empty state and flows through nodes
    result = graph.invoke({})
    return result.get("summary", "No summary generated")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "plan_now":
        summary = plan_now()
        print("\n===== PLAN SUMMARY =====\n" + summary)
    else:
        print("Commands:\n  plan_now   — run the planning flow once (last 24h)")
