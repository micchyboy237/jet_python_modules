from typing import TypedDict


class SwarmState(TypedDict):
    query: str
    subtasks: list[dict]
    findings: list[dict]
    iteration: int
    tokens_used: int
    start_time: float
    final_answer: str | None
