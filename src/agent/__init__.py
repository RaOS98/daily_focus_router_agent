from .workflow_graph import build_graph

def build_executor():
    """Build the LangGraph workflow executor (backward compatibility)"""
    return build_graph()
