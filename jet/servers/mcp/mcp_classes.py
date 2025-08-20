from typing import Dict, Any, TypedDict, TypeVar, Generic
from pydantic import BaseModel

# Generic type variables for input and output schemas
InputSchema = TypeVar("InputSchema", bound=TypedDict)
OutputSchema = TypeVar("OutputSchema", bound=TypedDict)


class ToolArguments(TypedDict, Generic[InputSchema]):
    arguments: InputSchema


class ToolInfo(TypedDict, Generic[InputSchema, OutputSchema]):
    name: str
    description: str
    schema: ToolArguments[InputSchema]
    outputSchema: OutputSchema


class ToolRequest(BaseModel):
    tool: str  # Changed from ToolInfo to str to match JSON input
    arguments: Dict[str, Any]


class ExecutedToolResponse(BaseModel):
    isError: bool
    meta: Dict[str, Any]
    content: Dict[str, Any]

    def __str__(self) -> str:
        return self.content.get("text", "")
