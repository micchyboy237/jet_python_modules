from typing import List, Dict, Optional, TypedDict, Union, Literal
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(...,
                      description="Role of the message sender (e.g., 'user', 'assistant')")
    content: str = Field(..., description="Content of the message")


class SystemMessage(Message):
    role: Literal['system'] = Field('system',
                                    description="Fixed role for system messages")


class AssistantMessage(Message):
    role: Literal['assistant'] = Field('assistant',
                                       description="Fixed role for assistant messages")


class UserMessage(Message):
    role: Literal['user'] = Field('user',
                                  description="Fixed role for user messages")


class Delta(BaseModel):
    role: Optional[str] = Field(
        None, description="Role of the message sender in streaming delta")
    content: Optional[str] = Field(
        None, description="Content of the message in streaming delta")


class BaseCompletionRequest(BaseModel):
    temperature: Optional[float] = Field(
        default=0.0, description="Sampling temperature", ge=0.0)
    top_p: Optional[float] = Field(
        default=1.0, description="Nucleus sampling parameter", ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(
        default=512, description="Maximum number of tokens to generate", ge=1)
    stream: Optional[bool] = Field(
        default=False, description="Whether to stream the response")
    stop: Optional[Union[str, List[str]]] = Field(
        default=None, description="Sequences where generation should stop")
    repetition_penalty: Optional[float] = Field(
        default=None, description="Penalty for repeated tokens", ge=1.0)
    repetition_context_size: Optional[int] = Field(
        default=20, description="Context window size for repetition penalty", ge=1)
    xtc_probability: Optional[float] = Field(
        default=0.0, description="Probability for XTC", ge=0.0, le=1.0)
    xtc_threshold: Optional[float] = Field(
        default=0.0, description="Threshold for XTC", ge=0.0)
    logit_bias: Optional[Dict[int, float]] = Field(
        default=None, description="Token ID to bias value mapping")
    logprobs: Optional[int] = Field(
        default=-1, description="Number of top tokens and log probabilities to return", ge=-1)
    model: Optional[str] = Field(
        default=None, description="Path to local model or Hugging Face repo ID")
    adapters: Optional[str] = Field(
        default=None, description="Path to low-rank adapters")
    draft_model: Optional[str] = Field(
        default=None, description="Smaller model for speculative decoding")
    num_draft_tokens: Optional[int] = Field(
        default=3, description="Number of draft tokens for draft model", ge=1)
    verbose: Optional[bool] = Field(
        default=False, description="Whether to enable verbose logging")
    worker_verbose: Optional[bool] = Field(
        default=False, description="Whether to enable verbose logging for workers")
    task_id: Optional[str] = Field(
        default=None, description="Unique identifier for the task")


class ChatCompletionRequest(BaseCompletionRequest):
    messages: Union[str, List[Message], List[List[Message]]] = Field(
        ..., description="String, array of message objects, or array of arrays of message objects representing conversation history")
    role_mapping: Optional[Dict[str, str]] = Field(
        default=None, description="Custom role prefixes for prompt generation")
    tools: Optional[List[Dict]] = Field(
        default=None, description="List of tools available for the model")
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt for the conversation")
    session_id: Optional[str] = Field(
        default=None, description="Unique identifier for the session")


class TextCompletionRequest(BaseCompletionRequest):
    prompt: Union[str, List[str]] = Field(
        ..., description="Input prompt or prompts for text completion")


class Usage(BaseModel):
    prompt_tokens: int = Field(...,
                               description="Number of prompt tokens processed")
    prompt_tps: float = Field(..., description="Prompt tokens per second")
    completion_tokens: int = Field(...,
                                   description="Number of tokens generated")
    completion_tps: float = Field(...,
                                  description="Completion tokens per second")
    total_tokens: int = Field(..., description="Total number of tokens")
    peak_memory: float = Field(..., description="Peak memory usage in MB")


class UnifiedCompletionResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the completion")
    created: int = Field(...,
                         description="Timestamp for when the request was processed")
    content: Optional[str] = Field(
        None, description="Generated content (text or message)")
    finish_reason: Optional[Literal["stop", "length"]] = Field(
        None, description="Reason the generation ended")
    usage: Optional[Usage] = Field(None, description="Token usage information")
    prompt_id: Optional[str] = Field(
        None, description="Unique identifier for the prompt")
    task_id: Optional[str] = Field(
        None, description="Unique identifier for the task")


class ModelInfo(BaseModel):
    id: str = Field(..., description="Hugging Face repo ID")
    short_name: str = Field(..., description="Model key")
    object: Optional[str] = Field(
        "model", description="Type of object, default is 'model'")
    created: int = Field(..., description="Timestamp for model creation")


class ModelsResponse(BaseModel):
    object: str = Field("list", description="Type of response, always 'list'")
    data: List[ModelInfo] = Field(..., description="List of available models")


class ParallelCompletionResponse(BaseModel):
    type: Literal["chunk", "result",
                  "error"] = Field(..., description="Type of response")
    prompt: Optional[str] = Field(
        None, description="Original prompt or message")
    content: Optional[str] = Field(None, description="Generated content")
    prompt_id: Optional[str] = Field(
        None, description="Unique identifier for the prompt")
    task_id: Optional[str] = Field(
        None, description="Unique identifier for the task")
    truncated: Optional[bool] = Field(
        None, description="Whether the response was truncated")
    message: Optional[str] = Field(
        None, description="Error message, if type is error")


class Logprobs(TypedDict):
    token_logprobs: List[float]
    top_logprobs: List[Dict[int, float]]
    tokens: List[int]


class Choice(BaseModel):
    index: int = Field(...,
                       description="Index of the choice in the response list")
    logprobs: Optional["Logprobs"] = Field(
        None, description="Log probabilities of the tokens")
    finish_reason: Optional[Literal["length", "stop"]] = Field(
        None, description="Reason the completion finished"
    )
    message: Optional[Message] = Field(
        None, description="Full message object, if available")
    delta: Optional[Delta] = Field(
        None, description="Delta message for streaming response")
    text: Optional[str] = Field(None, description="Generated text, if present")


class CompletionResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the completion")
    system_fingerprint: str = Field(...,
                                    description="System fingerprint for traceability")
    object: str = Field(...,
                        description="Type of object returned (e.g., 'text_completion')")
    model: str = Field(..., description="Name or path of the model used")
    created: int = Field(...,
                         description="Timestamp of when the response was generated")
    choices: List[Choice] = Field(...,
                                  description="List of generated completions or deltas")
    usage: Optional[Usage] = Field(
        None, description="Token usage and performance information")
