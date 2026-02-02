# callbacks.py
from memory.shared_state import shared_state
from smolagents import ActionStep, CodeAgent
from tools.memory_tools import LongTermSaveTool


def auto_save_shared_state(step: ActionStep, agent: CodeAgent) -> None:
    """Save shared state after steps that look like final answers or big updates"""
    if step.is_final_answer or "saved" in str(step.observations or "").lower():
        shared_state.save()


def auto_extract_simple_facts(step: ActionStep, agent: CodeAgent) -> None:
    """Very naive auto-extraction – improve with LLM reflection in production"""
    if not step.observations:
        return
    text = str(step.observations)
    if len(text) > 400:
        text = text[:380] + "..."
    keywords = ["important:", "remember:", "note that", "fact:", "key point"]
    if any(k in text.lower() for k in keywords):
        LongTermSaveTool().forward(content=text, step_number=step.step_number)
