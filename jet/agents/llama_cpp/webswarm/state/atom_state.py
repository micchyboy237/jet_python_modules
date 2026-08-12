from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AtomState(TypedDict):
    task: str  # Local objective
    messages: Annotated[list[BaseMessage], add_messages]  # ReAct conversation
    step_count: int  # Current step counter
    max_steps: int  # Step budget
    result: dict[str, Any] | None  # Final extracted answer
    is_complete: bool  # Termination flag
