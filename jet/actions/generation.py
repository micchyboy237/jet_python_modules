import re
import random
import threading
import json
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.utils import set_global_tokenizer
import requests
import traceback
from enum import Enum
from typing import Generator, Literal, Optional, TypedDict, Union
# from langchain_ollama.llms import OllamaLLM
from langchain_core.outputs.llm_result import LLMResult, GenerationChunk
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from jet.llm.llm_types import (
    Message,
    OllamaChatOptions,
    OllamaChatResponse,
    ChatResponseInfo,
    Tool,
    MessageRole,
    Track,
)
from jet.utils.class_utils import get_class_name
from jet.logger import logger
from jet.transformers import make_serializable
from shared.setup.events import EventSettings

DETERMINISTIC_LLM_SETTINGS = {
    # "seed": random.randint(0, 1000),
    "temperature": 0.3,
    "num_keep": 0,
    "num_predict": -1,
}

PROMPT_CONTEXT_TEMPLATE = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query}\n"
    "Answer: "
)


def sanitize_header_value(value: str) -> str:
    if not value:
        return ''

    value = re.sub(r'[\r\n]+', ' ', value).strip()
    # Replace non-ASCII characters with a safe fallback
    value = value.encode("ascii", errors="replace").decode("ascii")
    return value


def call_ollama_chat(
    messages: str | list[Message] | PromptTemplate,
    model: str = "llama3.2",
    *,
    system: Optional[str] = None,
    context: Optional[str] = None,
    tools: list[Tool] = None,
    format: Optional[Union[str, dict]] = None,
    options: OllamaChatOptions = {},
    stream: bool = True,
    keep_alive: Union[str, int] = "15m",
    template: Optional[str | PromptTemplate] = None,
    template_vars: dict = {},
    track: Track = None,
    full_stream_response: bool = False,
    max_tokens: Optional[int | float] = None,
    max_prediction_ratio: Optional[float] = None,
    buffer: int = 0,
    # Parameter for cancellation
    stop_event: Optional[threading.Event] = None,
) -> Union[str | OllamaChatResponse, Generator[str | OllamaChatResponse, None, None]]:
    """
    Wraps call_ollama_chat to track the prompt and response using Aim.

    Args:
        messages (str | list): The prompt or messages for the conversation.
        model (str | list[Message]): The name of the model to use.
        system (str): System message for the LLM.
        context (str): Context message for the LLM.
        tools (list): Tools for the model to use.
        format (Union[str, dict]): Format of the response ("json" or JSON schema).
        options (dict): Additional model parameters like temperature.
        stream (bool): Whether to stream the response (defaults to True).
        track (Track): Enable Aim tracking for prompt, response and other metadata.
        full_stream_response (bool): For stream only. Enable full dict format for each chunk.

    Returns:
         Union[str | OllamaChatResponse, Generator[str | OllamaChatResponse, None, None]]:
         Either the JSON response or a generator for streamed responses.
    """
    from jet.llm.models import OLLAMA_MODEL_EMBEDDING_TOKENS
    from jet.token.token_utils import get_ollama_tokenizer, filter_texts, token_counter, calculate_num_predict_ctx
    tokenizer = get_ollama_tokenizer(model)
    set_global_tokenizer(tokenizer)

    # logger.info("pre stop_event:", stop_event)
    # logger.orange(stop_event and stop_event.is_set())
    # if stop_event and stop_event.is_set():
    #     stop_event.clear()

    if track:
        from aim import Run, Text

    URL = "http://localhost:11434/api/chat"

    if isinstance(messages, str):
        messages = [
            Message(content=messages, role=MessageRole.USER)
        ]
    elif isinstance(messages, PromptTemplate) and not context:
        template = messages
        prompt = template.format(**template_vars)
        messages = [
            Message(content=prompt, role=MessageRole.USER)
        ]

    # Updates latest user message with context if available
    if context:
        # Iterate through messages from the end to find the latest user message
        latest_user_message: Optional[Message] = None
        for msg in reversed(messages):
            if msg["role"] == MessageRole.USER:
                latest_user_message = msg
                break

        if not latest_user_message:
            raise ValueError(
                "No user message found in the conversation history.")

        # Construct the context with the latest user message
        context = context if context else ""
        query = latest_user_message["content"]

        # Use the template if available, otherwise use the default prompt template
        if template:
            prompt = template.format(
                **template_vars, context=context, query=query)
        else:
            template = PROMPT_CONTEXT_TEMPLATE
            prompt = template.format(context=context, query=query)
        latest_user_message["content"] = prompt

    model_max_length = OLLAMA_MODEL_EMBEDDING_TOKENS[model]
    prompt_tokens: int = token_counter(messages, model)

    system = system or next(
        (m['content'] for m in messages if m['role'] == MessageRole.SYSTEM), None)
    system_tokens: int = 0
    if system:
        system_tokens = token_counter(system, model)

    logger.newline()
    logger.orange("Calling Ollama chat...")
    logger.log(
        "LLM model:",
        model,
        f"({model_max_length})",
        "|",
        "Tokens:",
        system_tokens + prompt_tokens,
        colors=["GRAY", "INFO", "INFO", "GRAY", "INFO", "INFO"],
    )
    logger.newline()

    if max_tokens:
        messages = filter_texts(
            messages, model, max_tokens=max_tokens)

    num_predict = options.get("num_predict", -1)
    num_ctx = model_max_length
    # if max_prediction_ratio and (not num_predict or num_predict <= 0):
    # calc_result = calculate_num_predict_ctx(messages, model, system=system)
    # num_predict = calc_result["num_predict"]
    # num_ctx = calc_result["num_ctx"]
    # else:
    # num_ctx = options.get("num_ctx", model_max_length)
    predict_tokens = num_ctx - (system_tokens + prompt_tokens)
    max_prompt_tokens = model_max_length - buffer
    derived_options = {
        "num_predict": num_predict,
        "num_ctx": num_ctx,
    }
    options = {
        **DETERMINISTIC_LLM_SETTINGS,
        **(options or {}),
        **derived_options,
    }

    logger.newline()
    logger.gray("LLM Settings:")
    for key, value in options.items():
        logger.log(f"{key}:", value, colors=["GRAY", "DEBUG"])

    logger.newline()
    logger.log("Stream:", stream, colors=["GRAY", "INFO"])
    logger.log("Model:", model, colors=["GRAY", "INFO"])
    logger.log("System Tokens:", system_tokens, colors=["GRAY", "DEBUG"])
    logger.log("Prompt Tokens:", prompt_tokens, colors=["GRAY", "DEBUG"])
    logger.log("Remaining Tokens:", predict_tokens, colors=["GRAY", "INFO"])
    logger.log("Max Prompt Tokens:", max_prompt_tokens,
               colors=["GRAY", "INFO"])
    logger.log("num_ctx:", num_ctx, colors=["GRAY", "ORANGE"])
    logger.log("Max Tokens:", model_max_length, colors=["GRAY", "ORANGE"])
    logger.newline()

    if system_tokens + prompt_tokens > max_prompt_tokens:
        raise ValueError(
            f"Token count ({system_tokens + prompt_tokens}) exceeds maximum allowed tokens ({max_prompt_tokens})"
        )

    logger.debug("Generating response...")

    if system and not any(m['role'] == MessageRole.SYSTEM for m in messages):
        messages = [m for m in messages if m['role'] != MessageRole.SYSTEM]
        messages.insert(0, Message(
            content=system, role=MessageRole.SYSTEM))

    if tools:
        stream = False
        options["temperature"] = 0

    # Format tool outputs to string if any
    messages = convert_tool_outputs_to_string(messages)

    # Prepare the request body
    body = {
        "model": model,
        "messages": messages,
        "stream": stream,
        # "template": template,
        # "keep_alive": keep_alive,
        # "raw": False,
        "tools": tools,
        # "format": str(format) if format else None,
        "format": format,
        "options": options,
    }
    body = make_serializable(body)

    # Initialize Aim run
    if track:
        run_settings: Track = {
            "log_system_params": True,
            **track.copy()
        }
        del run_settings["run_name"]
        del run_settings["metadata"]
        if "format" in run_settings:
            del run_settings["format"]

        run = Run(**run_settings)

    # Define headers
    event = EventSettings.call_ollama_chat()
    pre_start_hook_start_time = EventSettings.event_data["pre_start_hook"]["start_time"]
    # log_filename = f"{event['filename'].split(".")[0]}_{
    #     pre_start_hook_start_time}"
    log_filename = event['filename'].split(".")[0]
    logger.log("Log-Filename:", log_filename, colors=["WHITE", "DEBUG"])
    # Sanitize headers to remove invalid characters
    headers = {
        # Include the token count here
        "Tokens": str(system_tokens + prompt_tokens),
        "Log-Filename": sanitize_header_value(log_filename),
        "Event-Start-Time": sanitize_header_value(pre_start_hook_start_time),
        "System": sanitize_header_value(system),
        "Context": sanitize_header_value(context) if context else "",
        "Template": sanitize_header_value(template.template) if template else ""
    }

    try:
        # Make the POST request with headers
        r = requests.post(
            url=URL,
            json=body,
            headers=headers,
            stream=stream,
        )
        r.raise_for_status()

        if stream:
            response_chunks = []

            def line_generator():
                for line in r.iter_lines():
                    # logger.info("post stop_event:", stop_event)
                    # logger.orange(stop_event and stop_event.is_set())

                    if stop_event and stop_event.is_set():  # Check if stop event is triggered
                        logger.newline()
                        logger.warning(
                            "post stop_event: Streaming stopped by user.")
                        break

                    if line:
                        decoded_line = line.decode("utf-8")
                        try:
                            decoded_chunk: OllamaChatResponse = json.loads(
                                decoded_line)
                            content = decoded_chunk.get(
                                "message", {}).get("content", "")
                            response_chunks.append(content)
                            logger.success(content, flush=True)

                            if decoded_chunk.get("done"):
                                output = "".join(response_chunks)

                                # Calculate token counts
                                prompt_token_count: int = token_counter(
                                    messages, model)
                                response_token_count: int = token_counter(
                                    output, model)

                                decoded_chunk['prompt_eval_count'] = prompt_token_count
                                decoded_chunk['eval_count'] = response_token_count

                                response_info: ChatResponseInfo = decoded_chunk.copy()
                                response_info["options"] = options

                                logger.newline()
                                logger.newline()
                                logger.newline()

                                logger.log("Model:", model,
                                           colors=["WHITE", "DEBUG"])
                                logger.log("Options:", options,
                                           colors=["WHITE", "DEBUG"])
                                logger.log("Stream:", stream,
                                           colors=["WHITE", "DEBUG"])
                                logger.log("Response:", len(output),
                                           colors=["WHITE", "DEBUG"])
                                logger.log("Content:", len(
                                    str(messages)) + len(output), colors=["WHITE", "DEBUG"])

                                logger.newline()

                                # Get durations
                                # Print all duration values from decoded_chunk
                                durations = {
                                    k: v for k, v in response_info.items() if k.endswith('duration')}
                                if durations:
                                    logger.info("Durations:")
                                    for key, value in durations.items():
                                        # Convert nanoseconds to seconds/minutes/milliseconds
                                        seconds = value / 1e9
                                        if seconds >= 60:
                                            minutes = seconds / 60
                                            logger.log(f"{key}:", f"{minutes:.2f}m", colors=[
                                                "WHITE", "ORANGE"])
                                        elif seconds >= 1:
                                            logger.log(f"{key}:", f"{seconds:.2f}s", colors=[
                                                "WHITE", "WARNING"])
                                        else:
                                            millis = seconds * 1000
                                            logger.log(f"{key}:", f"{millis:.2f}ms", colors=[
                                                "WHITE", "LIME"])

                                logger.newline()
                                logger.newline()
                                logger.info("Final tokens info:")
                                # logger.success(json.dumps(
                                #     response_info, indent=2))
                                response_prompt_tokens = response_info['prompt_eval_count']
                                response_tokens = response_info['eval_count']
                                total_tokens = system_tokens + response_prompt_tokens + response_tokens
                                logger.log("System tokens:", system_tokens, colors=[
                                           "DEBUG", "SUCCESS"])
                                logger.log("Prompt tokens:", response_prompt_tokens, colors=[
                                           "DEBUG", "SUCCESS"])
                                logger.log("Response tokens:", response_tokens, colors=[
                                           "DEBUG", "SUCCESS"])
                                logger.log("Total tokens:", total_tokens, colors=[
                                           "DEBUG", "SUCCESS"])
                                logger.newline()

                                # For Aim tracking
                                if track:
                                    # Log the prompt (messages) to Aim
                                    prompt = messages[-1]['content']
                                    response = "".join(response_chunks)

                                    value = {
                                        "system": system,
                                        "prompt": prompt,
                                        "response": response,
                                    }

                                    formatted_value = json.dumps(
                                        value, indent=1)

                                    if track.get('format'):
                                        formatted_value = track['format'].format(
                                            **value)

                                    aim_value = Text(formatted_value)
                                    aim_context = {
                                        "model": model,
                                        "options": options,
                                        **track.get('metadata', {})
                                    }
                                    track_args = {
                                        "name": track['run_name'],
                                        "context": aim_context,
                                    }
                                    logger.newline()
                                    logger.log("Run Settings:", json.dumps(
                                        run_settings, indent=2), colors=["WHITE", "INFO"])
                                    logger.log("Aim Track:", json.dumps(
                                        track_args, indent=2), colors=["WHITE", "INFO"])
                                    run.track(aim_value, **track_args)

                            if full_stream_response:
                                yield decoded_chunk
                            else:
                                yield content

                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to decode JSON: {decoded_line}")

            return line_generator()
        else:
            response = r.json()
            response["options"] = options

            output = response["message"]["content"]
            logger.success(output)

            # Calculate token counts
            prompt_token_count: int = token_counter(messages, model)
            response_token_count: int = token_counter(output, model)

            response['prompt_eval_count'] = prompt_token_count
            response['eval_count'] = response_token_count

            response_info: ChatResponseInfo = response.copy()
            response_info["options"] = options

            logger.newline()
            logger.newline()

            logger.log("Model:", model,
                       colors=["WHITE", "DEBUG"])
            logger.log("Options:", options,
                       colors=["WHITE", "DEBUG"])
            logger.log("Stream:", stream,
                       colors=["WHITE", "DEBUG"])
            logger.log("Response:", len(output),
                       colors=["WHITE", "DEBUG"])
            logger.log("Content:", len(
                str(messages)) + len(output), colors=["WHITE", "DEBUG"])

            logger.newline()

            # Get durations
            # Print all duration values from decoded_chunk
            durations = {
                k: v for k, v in response_info.items() if k.endswith('duration')}
            if durations:
                logger.info("Durations:")
                for key, value in durations.items():
                    # Convert nanoseconds to seconds/minutes/milliseconds
                    seconds = value / 1e9
                    if seconds >= 60:
                        minutes = seconds / 60
                        logger.log(f"{key}:", f"{minutes:.2f}m", colors=[
                            "WHITE", "ORANGE"])
                    elif seconds >= 1:
                        logger.log(f"{key}:", f"{seconds:.2f}s", colors=[
                            "WHITE", "WARNING"])
                    else:
                        millis = seconds * 1000
                        logger.log(f"{key}:", f"{millis:.2f}ms", colors=[
                            "WHITE", "LIME"])

            logger.newline()
            logger.info("Final tokens info:")
            # logger.success(json.dumps(response_info, indent=2))
            response_prompt_tokens = response_info['prompt_eval_count']
            response_tokens = response_info['eval_count']
            total_tokens = response_prompt_tokens + response_tokens
            logger.log("System tokens:", system_tokens,
                       colors=["DEBUG", "SUCCESS"])
            logger.log("Prompt tokens:", response_prompt_tokens,
                       colors=["DEBUG", "SUCCESS"])
            logger.log("Response tokens:", response_tokens,
                       colors=["DEBUG", "SUCCESS"])
            logger.log("Total tokens:", total_tokens,
                       colors=["DEBUG", "SUCCESS"])
            logger.newline()

            return response

    except requests.RequestException as e:
        logger.error(f"Request failed - {get_class_name(e)}: {e}")
        traceback.print_exc()
        if track:
            run.track({"error": str(e)}, name="error",
                      context={"model": model})
        return {"error": str(e)}

    except Exception as e:
        logger.error(f"Error class name: {get_class_name(e)}")
        traceback.print_exc()
        return {"error": str(e)}

    # finally:
    #     if track:
    #         run.close()


def convert_tool_outputs_to_string(ollama_messages: list[Message]):
    for message in ollama_messages:
        if message.get("role") == "tool" and not isinstance(message.get("content"), str):
            message["content"] = str(message["content"])
    return ollama_messages


# Main function to demonstrate sample usage
if __name__ == "__main__":
    model = "gemma2:9b"
    prompt = "Write a 20 word creative story about an explorer finding a hidden treasure."

    # No stream
    logger.newline()
    logger.info("No stream response:")
    response = call_ollama_chat(prompt, model, stream=False)

    # With stream
    logger.newline()
    logger.info("With stream response:")
    response = ""
    for chunk in call_ollama_chat(prompt, model):
        response += chunk
