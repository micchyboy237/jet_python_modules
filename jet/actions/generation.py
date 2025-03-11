import threading
import json
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
    "seed": 0,
    "temperature": 0,
    "num_keep": 0,
    "num_predict": -1,
}


def call_ollama_chat(
    messages: str | list[Message],
    model: str = "llama3.1",
    *,
    system: str = "",
    tools: list[Tool] = None,
    format: Union[str, dict] = None,
    options: OllamaChatOptions = {},
    stream: bool = True,
    keep_alive: Union[str, int] = "15m",
    template: str = None,
    track: Track = None,
    full_stream_response: bool = False,
    max_tokens: Optional[int | float] = None,
    max_prediction_ratio: Optional[float] = None,
    # Parameter for cancellation
    stop_event: Optional[threading.Event] = None,
) -> Union[str | OllamaChatResponse, Generator[str | OllamaChatResponse, None, None]]:
    """
    Wraps call_ollama_chat to track the prompt and response using Aim.

    Args:
        messages (str | list): The prompt or messages for the conversation.
        model (str | list[Message]): The name of the model to use.
        system (str): System message for the LLM.
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

    model_max_length = OLLAMA_MODEL_EMBEDDING_TOKENS[model]
    token_count: int = token_counter(messages, model)

    logger.newline()
    logger.orange("Calling Ollama chat...")
    logger.log(
        "LLM model:",
        model,
        f"({model_max_length})",
        "|",
        "Tokens:",
        token_count,
        colors=["GRAY", "INFO", "INFO", "GRAY", "INFO", "INFO"],
    )
    logger.newline()

    if max_tokens:
        messages = filter_texts(
            messages, model, max_tokens=max_tokens)

    num_predict = options.get("num_predict", -1)
    if max_prediction_ratio and (not num_predict or num_predict <= 0):
        calc_result = calculate_num_predict_ctx(messages, model, system=system)
        num_predict = calc_result["num_predict"]
        num_ctx = calc_result["num_ctx"]

        predict_tokens = num_predict
    else:
        num_ctx = options.get("num_ctx", model_max_length)
        predict_tokens = num_ctx - token_count
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
    logger.log("Prompt Tokens:", token_count, colors=["GRAY", "ORANGE"])
    logger.log("Remaining Tokens:", predict_tokens, colors=["GRAY", "ORANGE"])
    logger.log("num_ctx:", num_ctx, colors=["GRAY", "ORANGE"])
    logger.log("Max Tokens:", model_max_length, colors=["GRAY", "ORANGE"])
    logger.newline()

    logger.debug("Generating response...")

    if not any(message['role'] == MessageRole.USER for message in messages):
        messages[-1]["role"] = MessageRole.USER

    if system and not any(message['role'] == MessageRole.SYSTEM for message in messages):
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
        # "keep_alive": keep_alive,
        "template": template,
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
    headers = {
        "Tokens": str(token_count),  # Include the token count here
        "Log-Filename": log_filename,
        "Event-Start-Time": pre_start_hook_start_time,
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
                                prompt_tokens = response_info['prompt_eval_count']
                                response_tokens = response_info['eval_count']
                                total_tokens = prompt_tokens + response_tokens
                                logger.log("Prompt tokens:", prompt_tokens, colors=[
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
            prompt_tokens = response_info['prompt_eval_count']
            response_tokens = response_info['eval_count']
            total_tokens = prompt_tokens + response_tokens
            logger.log("Prompt tokens:", prompt_tokens,
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
