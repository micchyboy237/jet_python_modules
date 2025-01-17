from dis import Positions
from typing import TypedDict


class EventData(TypedDict, total=False):
    filepath: str
    function: str
    lineno: int
    code_context: list[str] | None
