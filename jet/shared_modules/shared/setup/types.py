from typing import Any, TypedDict


class BaseEventData(TypedDict, total=False):
    filepath: str
    filename: str
    function: str
    lineno: int
    code_context: list[str]


class OrigEventDataFunction(BaseEventData, total=False):
    pass


class EventData(BaseEventData):
    event_name: str
    orig_function: OrigEventDataFunction
    arguments: dict[str, Any]
    start_time: str
