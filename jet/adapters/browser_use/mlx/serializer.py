from typing import List
from jet.llm.mlx.remote.types import Message as MLXMessage
from browser_use.llm.messages import (
    AssistantMessage,
    BaseMessage,
    SystemMessage,
    UserMessage,
    ToolCall as BrowserUseToolCall,
)
from jet.llm.mlx.remote.types import ToolCall as MLXToolCall


class MLXMessageSerializer:
    """Serializer for converting between browser_use message types and MLX message types."""

    @staticmethod
    def _serialize_tool_calls(tool_calls: List[BrowserUseToolCall]) -> List[MLXToolCall]:
        """Convert browser_use ToolCalls to MLX ToolCalls."""
        mlx_tool_calls: List[MLXToolCall] = []
        for tool_call in tool_calls:
            arguments = tool_call.function.arguments
            # Ensure arguments is a dict, handle string case
            if isinstance(arguments, str):
                try:
                    import json
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"arguments": arguments}
            mlx_tool_call = {
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": arguments
                }
            }
            mlx_tool_calls.append(mlx_tool_call)
        return mlx_tool_calls

    @staticmethod
    def serialize(message: BaseMessage) -> MLXMessage:
        """Serialize a browser_use message to an MLX Message."""
        if isinstance(message, UserMessage):
            content = message.content if isinstance(message.content, str) else "".join(
                part.text for part in message.content if hasattr(part, 'type') and part.type == 'text'
            )
            return {
                "role": "user",
                "content": content or None,
                "tool_calls": None
            }
        elif isinstance(message, SystemMessage):
            content = message.content if isinstance(message.content, str) else "".join(
                part.text for part in message.content if hasattr(part, 'type') and part.type == 'text'
            )
            return {
                "role": "system",
                "content": content or None,
                "tool_calls": None
            }
        elif isinstance(message, AssistantMessage):
            content = message.content if isinstance(message.content, str) else "".join(
                part.text for part in message.content if hasattr(part, 'type') and part.type == 'text'
            )
            tool_calls = MLXMessageSerializer._serialize_tool_calls(
                message.tool_calls) if message.tool_calls else None
            return {
                "role": "assistant",
                "content": content or None,
                "tool_calls": tool_calls
            }
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

    @staticmethod
    def serialize_messages(messages: List[BaseMessage]) -> List[MLXMessage]:
        """Serialize a list of browser_use messages to MLX Messages."""
        return [MLXMessageSerializer.serialize(m) for m in messages]
