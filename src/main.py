from __future__ import annotations
import sys
from .agent import build_executor

from .agent.react_agent import REACT_INSTRUCTIONS

def plan_now() -> str:
    executor = build_executor()
    result = executor.invoke({"input": REACT_INSTRUCTIONS})
    return result["output"]

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "plan_now":
        summary = plan_now()
        print("\n===== PLAN SUMMARY =====\n" + summary)
    else:
        print("Commands:\n  plan_now   — run the planning flow once (last 24h)")
