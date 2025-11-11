import json
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.dataclasses.streaming_chunk import ComponentInfo, FinishReason, StreamingChunk, ToolCallDelta
from haystack.tools import (
    Tool,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    serialize_tools_or_toolset,
)
from haystack.tools.toolset import Toolset
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable
from pydantic.json_schema import JsonSchemaValue

from ollama import ChatResponse, Client

import os
from jet.llm.mlx.config import DEFAULT_OLLAMA_LOG_DIR
from jet.llm.logger_utils import ChatLogger
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.text import format_sub_dir

FINISH_REASON_MAPPING: Dict[str, FinishReason] = {
    "stop": "stop",
    "tool_calls": "tool_calls",
    # we skip load and unload reasons
}


def _convert_chatmessage_to_ollama_format(message: ChatMessage) -> Dict[str, Any]:
    """
    Convert a ChatMessage to the format expected by the Ollama Chat API.
    """
    text_contents = message.texts
    tool_calls = message.tool_calls
    tool_call_results = message.tool_call_results
    images = message.images

    if not text_contents and not tool_calls and not tool_call_results and not images:
        msg = (
            "A `ChatMessage` must contain at least one `TextContent`, `ToolCall`, `ToolCallResult`, or `ImageContent`."
        )
        raise ValueError(msg)
    elif len(text_contents) + len(tool_call_results) > 1:
        msg = "For Ollama compatibility, a `ChatMessage` can contain at most one `TextContent` or `ToolCallResult`."
        raise ValueError(msg)

    ollama_msg: Dict[str, Any] = {"role": message.role.value}

    if tool_call_results:
        # Ollama does not provide a way to communicate errors in tool invocations, so we ignore the error field
        ollama_msg["content"] = tool_call_results[0].result
        return ollama_msg

    if text_contents:
        ollama_msg["content"] = text_contents[0]
    if images:
        ollama_msg["images"] = [image.base64_image for image in images]
    if tool_calls:
        # Ollama does not support tool call id, so we ignore it
        ollama_msg["tool_calls"] = [
            {"type": "function", "function": {"name": tool_call.tool_name, "arguments": tool_call.arguments}}
            for tool_call in tool_calls
        ]
    if message.reasoning:
        ollama_msg["thinking"] = message.reasoning.reasoning_text
    return ollama_msg


def _convert_ollama_meta_to_openai_format(input_response_dict: Dict) -> Dict[str, Any]:
    """
    Map Ollama metadata keys onto the OpenAI-compatible names Haystack expects.
    All fields that are not part of the OpenAI metadata are left unchanged in the returned dict.

    Example Ollama metadata:
    {
        'model': 'phi4:14b-q4_K_M',
        'created_at': '2025-03-09T18:38:33.004185821Z',
        'done': True,
        'done_reason': 'stop',
        'total_duration': 86627206961,
        'load_duration': 23585622554,
        'prompt_eval_count': 26,
        'prompt_eval_duration': 3426000000,
        'eval_count': 298,
        'eval_duration': 4799921000
    }
    Example OpenAI metadata:
    {
        'model': 'phi4:14b-q4_K_M',
        'finish_reason': 'stop',
        'usage': {
            'completion_tokens': 298,
            'prompt_tokens': 26,
            'total_tokens': 324,
        }
        'completion_start_time': '2025-03-09T18:38:33.004185821Z',
        'done': True,
        'total_duration': 86627206961,
        'load_duration': 23585622554,
        'prompt_eval_duration': 3426000000,
        'eval_duration': 4799921000,
    }
    """
    meta = {key: value for key, value in input_response_dict.items() if key != "message"}

    if "done_reason" in meta:
        meta["finish_reason"] = FINISH_REASON_MAPPING.get(meta.pop("done_reason") or "")
    if "created_at" in meta:
        meta["completion_start_time"] = meta.pop("created_at")
    if "eval_count" in meta and "prompt_eval_count" in meta:
        eval_count = meta.pop("eval_count")
        prompt_eval_count = meta.pop("prompt_eval_count")
        meta["usage"] = {
            "completion_tokens": eval_count,
            "prompt_tokens": prompt_eval_count,
            "total_tokens": eval_count + prompt_eval_count,
        }
    return meta


def _convert_ollama_response_to_chatmessage(ollama_response: ChatResponse) -> ChatMessage:
    """
    Convert non-streaming Ollama Chat API response to Haystack ChatMessage with the assistant role.
    """
    response_dict = ollama_response.model_dump()
    ollama_message = response_dict["message"]
    text = ollama_message["content"]
    tool_calls: List[ToolCall] = []

    if ollama_tool_calls := ollama_message.get("tool_calls"):
        for ollama_tc in ollama_tool_calls:
            tool_calls.append(
                ToolCall(
                    tool_name=ollama_tc["function"]["name"],
                    arguments=ollama_tc["function"]["arguments"],
                )
            )

    reasoning = ollama_message.get("thinking", None)

    chat_msg = ChatMessage.from_assistant(text=text or None, tool_calls=tool_calls, reasoning=reasoning)

    chat_msg._meta = _convert_ollama_meta_to_openai_format(response_dict)

    return chat_msg


def _build_chunk(
    chunk_response: ChatResponse, component_info: ComponentInfo, index: int, tool_call_index: int
) -> StreamingChunk:
    """
    Convert one Ollama stream-chunk to Haystack StreamingChunk.
    """
    chunk_response_dict = chunk_response.model_dump()
    finish_reason = FINISH_REASON_MAPPING.get(chunk_response.done_reason or "")
    tool_calls_list = []

    content = chunk_response_dict["message"]["content"]

    meta = {key: value for key, value in chunk_response_dict.items() if key != "message"}
    meta["role"] = chunk_response_dict["message"]["role"]

    # until a specific field in StreamingChunk is available, we store the thinking in the meta
    meta["reasoning"] = chunk_response_dict["message"].get("thinking", None)

    if tool_calls := chunk_response_dict["message"].get("tool_calls"):
        for tool_call in tool_calls:
            tool_calls_list.append(
                ToolCallDelta(
                    index=tool_call_index,
                    tool_name=tool_call["function"]["name"],
                    arguments=json.dumps(tool_call["function"]["arguments"])
                    if tool_call["function"]["arguments"]
                    else "",
                )
            )

    return StreamingChunk(
        content=content,
        meta=meta,
        index=index,
        finish_reason=finish_reason,
        component_info=component_info,
        tool_calls=tool_calls_list,
    )


@component
class OllamaChatGenerator:
    """
    Haystack Chat Generator for models served with Ollama (https://ollama.ai).

    Supports streaming, tool calls, reasoning, and structured outputs.

    Usage example:
    ```python
    from haystack_integrations.components.generators.ollama.chat import OllamaChatGenerator
    from haystack.dataclasses import ChatMessage

    llm = OllamaChatGenerator(model="qwen3:0.6b")
    result = llm.run(messages=[ChatMessage.from_user("What is the capital of France?")])
    print(result)
    ```
    """

    def __init__(
        self,
        model: str = "qwen3:0.6b",
        url: str = "http://localhost:11434",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
        keep_alive: Optional[Union[float, str]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        tools: Optional[Union[List[Tool], Toolset]] = None,
        response_format: Optional[Union[None, Literal["json"], JsonSchemaValue]] = None,
        think: Union[bool, Literal["low", "medium", "high"]] = False,
        log_dir: Optional[str] = None,
        agent_name: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        :param model:
            The name of the model to use. The model must already be present (pulled) in the running Ollama instance.
        :param url:
            The base URL of the Ollama server (default "http://localhost:11434").
        :param generation_kwargs:
            Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, and others. See the available arguments in
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :param timeout:
            The number of seconds before throwing a timeout error from the Ollama API.
        :param think
            If True, the model will "think" before producing a response.
            Only [thinking models](https://ollama.com/search?c=thinking) support this feature.
            Some models like gpt-oss support different levels of thinking: "low", "medium", "high".
            The intermediate "thinking" output can be found by inspecting the `reasoning` property of the returned
            `ChatMessage`.
        :param keep_alive:
            The option that controls how long the model will stay loaded into memory following the request.
            If not set, it will use the default value from the Ollama (5 minutes).
            The value can be set to:
            - a duration string (such as "10m" or "24h")
            - a number in seconds (such as 3600)
            - any negative number which will keep the model loaded in memory (e.g. -1 or "-1m")
            - '0' which will unload the model immediately after generating a response.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param tools:
            A list of `haystack.tools.Tool` or a `haystack.tools.Toolset`. Duplicate tool names raise a `ValueError`.
            Not all models support tools. For a list of models compatible with tools, see the
            [models page](https://ollama.com/search?c=tools).
        :param response_format:
            The format for structured model outputs. The value can be:
            - None: No specific structure or format is applied to the response. The response is returned as-is.
            - "json": The response is formatted as a JSON object.
            - JSON Schema: The response is formatted as a JSON object
                that adheres to the specified JSON Schema. (needs Ollama ≥ 0.1.34)
        """
        _check_duplicate_tool_names(list(tools or []))

        self.model = model
        self.url = url
        self.generation_kwargs = generation_kwargs or {}
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.streaming_callback = streaming_callback
        self.tools = tools
        self.think = think
        self.response_format = response_format

        self._client = Client(host=self.url, timeout=self.timeout)

        log_dir = os.path.join(DEFAULT_OLLAMA_LOG_DIR, log_dir or "")
        if agent_name:
            log_dir += f"/{format_sub_dir(agent_name)}"
        self._chat_logger = ChatLogger(log_dir, method="chat")
        self.verbose = verbose

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
              Dictionary with serialized data.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None

        return default_to_dict(
            self,
            model=self.model,
            url=self.url,
            generation_kwargs=self.generation_kwargs,
            timeout=self.timeout,
            keep_alive=self.keep_alive,
            streaming_callback=callback_name,
            tools=serialize_tools_or_toolset(self.tools),
            response_format=self.response_format,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OllamaChatGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.

        :returns:
            Deserialized component.
        """
        deserialize_tools_or_toolset_inplace(data["init_parameters"], key="tools")
        if callback_ser := data["init_parameters"].get("streaming_callback"):
            data["init_parameters"]["streaming_callback"] = deserialize_callable(callback_ser)
        return default_from_dict(cls, data)

    def _handle_streaming_response(
        self,
        response_iter: Iterator[ChatResponse],
        callback: Optional[Callable[[StreamingChunk], None]],
    ) -> Dict[str, List[ChatMessage]]:
        """
        Process streaming responses from the Ollama chat model and build a ChatMessage.

        This method iterates over streaming chunks from the Ollama model, constructs StreamingChunk objects,
        handles tool calls, and aggregates the content into a final ChatMessage. It supports verbose logging
        and callbacks for real-time streaming output. The method is designed to be robust against JSON parsing
        errors and async worker failures, ensuring graceful error handling.

        Args:
            response_iter (Iterator[ChatResponse]): Iterator yielding ChatResponse objects from the Ollama model.
            callback (Optional[Callable[[StreamingChunk], None]]): Optional callback to process each StreamingChunk
                in real time, useful for streaming output to a client or UI.

        Returns:
            Dict[str, List[ChatMessage]]: A dictionary containing a single key "replies" with a list of one
                ChatMessage, which includes the aggregated text, tool calls, reasoning, and metadata.

        Raises:
            Exception: If an error occurs while processing the streaming response, it is logged with a stack trace
                and re-raised to allow upstream handling.
        """
        # Initialize component metadata and data structures for processing chunks
        component_info = ComponentInfo.from_component(self)
        chunks: List[StreamingChunk] = []
        arg_by_id: Dict[str, str] = {}  # Maps tool call IDs to their arguments
        name_by_id: Dict[str, str] = {}  # Maps tool call IDs to their tool names
        id_order: List[str] = []  # Tracks the order of tool call IDs
        tool_call_index: int = 0  # Increments for each tool call to track sequence

        # Process streaming chunks with error handling for async issues
        try:
            for index, raw in enumerate(response_iter):
                if self.verbose and raw.done:
                    # Log raw chunk for debugging
                    logger.debug(f"\n\nRaw chunk response: {raw.model_dump()}")
                # Increment tool call index if the chunk contains tool calls
                if raw.message.tool_calls:
                    tool_call_index += 1
                # Build a StreamingChunk from the raw response
                chunk = _build_chunk(
                    chunk_response=raw,
                    component_info=component_info,
                    index=index,
                    tool_call_index=tool_call_index
                )
                chunks.append(chunk)
                # Log chunk content in teal if verbose mode is enabled
                if self.verbose:
                    logger.teal(chunk.content, flush=True)
                # Mark the chunk as the start if it's the first chunk or contains tool calls
                start = index == 0 or bool(chunk.tool_calls)
                chunk.start = start
                # Process tool calls in the chunk
                if chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        tool_call_id = tool_call.id or tool_call.tool_name or ""
                        args = tool_call.arguments or ""
                        # Store new tool call IDs and their names/arguments
                        if tool_call_id not in id_order:
                            id_order.append(tool_call_id)
                            name_by_id[tool_call_id] = tool_call.tool_name or ""
                        arg_by_id[tool_call_id] = args
                # Invoke callback for real-time streaming if provided
                if callback:
                    callback(chunk)
        except Exception as e:
            # Log any async or processing errors with full stack trace
            logger.error(f"Error processing streaming response: {e}", exc_info=True)
            raise

        # Aggregate text and reasoning from all chunks
        text = ""
        reasoning = ""
        for c in chunks:
            text += c.content
            reasoning += c.meta.get("reasoning", None) or ""

        # Build tool calls from collected IDs and arguments
        tool_calls: List[ToolCall] = []
        for tool_call_id in id_order:
            arguments = arg_by_id.get(tool_call_id, "")
            try:
                # Parse arguments as JSON, default to empty dict if empty or invalid
                parsed_args = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                parsed_args = {}
            tool_calls.append(ToolCall(tool_name=name_by_id[tool_call_id], arguments=parsed_args))

        # Create the final ChatMessage with aggregated data
        reply = ChatMessage.from_assistant(
            text=text or None,
            tool_calls=tool_calls or None,
            reasoning=reasoning or None,
            meta=_convert_ollama_meta_to_openai_format(chunks[-1].meta) if chunks else {},
        )
        return {"replies": [reply]}

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[Union[List[Tool], Toolset]] = None,
        *,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ) -> Dict[str, List[ChatMessage]]:
        """
        Runs an Ollama Model on a given chat history.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param generation_kwargs:
            Per-call overrides for Ollama inference options.
            These are merged on top of the instance-level `generation_kwargs`.
            Optional arguments to pass to the Ollama generation endpoint, such as temperature, top_p, etc. See the
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :param tools:
            A list of tools or a Toolset for which the model can prepare calls. This parameter can accept either a
            list of `Tool` objects or a `Toolset` instance. If set, it will override the `tools` parameter set
            during component initialization.
        :param streaming_callback:
            A callable to receive `StreamingChunk` objects as they
            arrive.  Supplying a callback (here or in the constructor) switches
            the component into streaming mode.
        :returns: A dictionary with the following keys:
            - `replies`: A list of ChatMessages containing the model's response
        """
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        tools = tools or self.tools
        _check_duplicate_tool_names(list(tools or []))

        # Convert Toolset → list[Tool] for JSON serialization
        if isinstance(tools, Toolset):
            tools = list(tools)
        ollama_tools = [{"type": "function", "function": {**tool.tool_spec}} for tool in tools] if tools else None

        callback = streaming_callback or self.streaming_callback
        # is_stream = callback is not None
        is_stream = True

        ollama_messages = [_convert_chatmessage_to_ollama_format(m) for m in messages]

        if self.verbose:
            logger.gray("Ollama Generator Settings:")
            logger.info(format_json({
                "messages": ollama_messages,
                "generation_kwargs": generation_kwargs,
                "stream": is_stream,
                "tools": ollama_tools,
            }))

        response = self._client.chat(
            model=self.model,
            messages=ollama_messages,
            tools=ollama_tools,
            stream=is_stream,  # type: ignore[call-overload]  # Ollama expects Literal[True] or Literal[False], not bool
            keep_alive=self.keep_alive,
            options=generation_kwargs,
            format=self.response_format,
            think=self.think,
        )

        if isinstance(response, Iterator):
            response = self._handle_streaming_response(response_iter=response, callback=callback)
        else:
            # non-stream path
            response = {"replies": [_convert_ollama_response_to_chatmessage(ollama_response=response)]}

        self._chat_logger.log_interaction(
            messages=ollama_messages,
            response=response["replies"][0]._content,
            model=self.model,
            url=self.url,
            stream=is_stream,
            think=self.think,
            keep_alive=self.keep_alive,
            options=generation_kwargs,
            tools=ollama_tools,
            response_meta=response["replies"][0]._meta
        )

        return response
