from typing import Literal, Union, Dict, List, Optional, TypedDict, Any, get_args
from enum import Enum
from jet.llm.mlx.helpers.detect_repetition import NgramRepeat
from jet.models.model_types import (
    ChatRole,
    LLMModelType,
    MLXTokenizer,
    Message,
    Delta,
    Tool,
    RoleMapping,
    Logprobs,
    Choice,
    Usage,
    ModelInfo,
    ModelsResponse,
    CompletionResponse,
)
from mlx_lm.tokenizer_utils import TokenizerWrapper


class DBMessage(Message):
    id: str
    session_id: str
    message_order: int
    updated_at: Optional[str]
    created_at: Optional[str]


class ChatTemplateArgs(TypedDict, total=False):
    """Configuration for chat template processing.

    Defines optional parameters for the `apply_chat_template` method, controlling
    how messages are formatted into prompts. All fields are optional to allow
    flexible customization while maintaining type safety.

    Fields:
        add_generation_prompt: Determines whether to append a generation prompt (e.g., assistant response marker).
        enable_thinking: Switches between thinking and non-thinking modes for the model.
        tokenize: Controls whether the output is tokenized (True) or returned as a string (False).
        add_special_tokens: Specifies whether special tokens (e.g., BOS, EOS) are included in the tokenized output.
        truncation: Enables truncation of the input to fit within the maximum length.
        max_length: Sets the maximum length for the tokenized prompt.
        include_system_prompt: Controls whether system messages are included in the prompt.
        tool_choice: Specifies whether to automatically select a tool ("auto") or disable tool usage ("none").
    """
    add_generation_prompt: bool
    enable_thinking: bool
    tokenize: bool
    add_special_tokens: bool
    truncation: bool
    max_length: int
    include_system_prompt: bool
    tool_choice: Literal["auto", "none"]


class ToolArguments(TypedDict):
    name: str
    arguments: Dict[str, Any]


class ToolCall(TypedDict):
    function: ToolArguments
    type: Literal["function"]


class ToolCallResult(TypedDict):
    tool_call: ToolCall
    tool_result: Any
