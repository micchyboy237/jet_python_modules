"""Convert messages between AutoGen v0.2 and v0.4 formats.

This module provides functions to convert messages between AutoGen v0.2 (`ChatResult.chat_history`) and v0.4 (`TaskResult.messages`) formats. It supports text, multi-modal, tool call, and tool response messages, enabling interoperability between the two versions for chat history and tool interactions.
"""

from typing import Any, Dict, List, Literal
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    HandoffMessage,
    MultiModalMessage,
    StopMessage,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_core import FunctionCall, Image
from autogen_core.models import FunctionExecutionResult

def convert_to_v02_message(
    message: BaseAgentEvent | BaseChatMessage,
    role: Literal["assistant", "user", "tool"],
    image_detail: Literal["auto", "high", "low"] = "auto",
) -> Dict[str, Any]:
    v02_message: Dict[str, Any] = {}
    if isinstance(message, TextMessage | StopMessage | H

andoffMessage | ToolCallSummaryMessage):
        v02_message = {"content": message.content, "role": role, "name": message.source}
    elif isinstance(message, MultiModalMessage):
        v02_message = {"content": [], "role": role, "name": message.source}
        for modal in message.content:
            if isinstance(modal, str):
                v02_message["content"].append({"type": "text", "text": modal})
            elif isinstance(modal, Image):
                v02_message["content"].append(modal.to_openai_format(detail=image_detail))
            else:
                raise ValueError(f"Invalid multimodal message content: {modal}")
    elif isinstance(message, ToolCallRequestEvent):
        v02_message = {"tool_calls": [], "role": "assistant", "content": None, "name": message.source}
        for tool_call in message.content:
            v02_message["tool_calls"].append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {"name": tool_call.name, "args": tool_call.arguments},
                }
            )
    elif isinstance(message, ToolCallExecutionEvent):
        tool_responses: List[Dict[str, str]] = []
        for tool_result in message.content:
            tool_responses.append(
                {
                    "tool_call_id": tool_result.call_id,
                    "role": "tool",
                    "content": tool_result.content,
                }
            )
        content = "\n\n".join([response["content"] for response in tool_responses])
        v02_message = {"tool_responses": tool_responses, "role": "tool", "content": content}
    else:
        raise ValueError(f"Invalid message type: {type(message)}")
    return v02_message

def convert_to_v04_message(message: Dict[str, Any]) -> BaseAgentEvent | BaseChatMessage:
    if "tool_calls" in message:
        tool_calls: List[FunctionCall] = []
        for tool_call in message["tool_calls"]:
            tool_calls.append(
                FunctionCall(
                    id=tool_call["id"],
                    name=tool_call["function"]["name"],
                    arguments=tool_call["function"]["args"],
                )
            )
        return ToolCallRequestEvent(source=message["name"], content=tool_calls)
    elif "tool_responses" in message:
        tool_results: List[FunctionExecutionResult] = []
        for tool_response in message["tool_responses"]:
            tool_results.append(
                FunctionExecutionResult(
                    call_id=tool_response["tool_call_id"],
                    content=tool_response["content"],
                    is_error=False,
                    name=tool_response["name"],
                )
            )
        return ToolCallExecutionEvent(source="tools", content=tool_results)
    elif isinstance(message["content"], list):
        content: List[str | Image] = []
        for modal in message["content"]:
            if modal["type"] == "text":
                content.append(modal["text"])
            else:
                content.append(Image.from_uri(modal["image_url"]["url"]))
        return MultiModalMessage(content=content, source=message["name"])
    elif isinstance(message["content"], str):
        return TextMessage(content=message["content"], source=message["name"])
    else:
        raise ValueError(f"Unable to convert message: {message}")