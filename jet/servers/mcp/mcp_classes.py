from typing import Dict, Any
from pydantic import BaseModel


class ToolRequest(BaseModel):
    tool: str
    arguments: Dict[str, Any]
