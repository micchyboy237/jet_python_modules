import json
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
from jet.utils import get_class_name
from jet.logger import logger
from jet.transformers import make_serializable
from jet.llm.token import token_counter

DEFAULT_SETTINGS: OllamaChatOptions = {
    "seed": 42,
    "temperature": 0,
    # "num_keep": 0,
    "num_predict": -1,
}

# class CustomStreamingHandler(StreamingStdOutCallbackHandler):
#     """Custom callback handler to log each new token with `logger.success`."""

#     def on_llm_new_token(self, token: str, **kwargs: dict) -> None:
#         logger.success(token, end="", flush=True)


# class Ollama:

#     def __init__(self, model: str = "mistral", base_url: str = "http://jetairm1:11434"):
#         self.model = model
#         self.base_url = base_url
#         self.ollama = OllamaLLM(model=model, base_url=base_url)

#     def generate(self, prompt: str, settings: dict[str, any] = None, raw: bool = False) -> dict[str, any]:
#         # Merge default settings with user-provided settings
#         settings = {**DEFAULT_SETTINGS, **(settings or {})}

#         data: LLMResult = self.ollama.generate(
#             prompts=[prompt],
#             options={"stream": True, "raw": raw, **settings},
#             # callbacks=[CustomStreamingHandler()],
#         )
#         generated_chunk: GenerationChunk = data.generations[0][0]
#         output = generated_chunk.text.strip()

#         return {
#             "prompt": prompt,
#             "output": output,
#             "meta": {
#                 "prompt_len": len(prompt),
#                 "output_len": len(output),
#                 "total_len": len(prompt) + len(output),
#             },
#             "settings": settings,
#         }


def call_ollama_chat(
    messages: str | list[Message],
    model: str = "llama3.1",
    *,
    system: str = "",
    tools: list[Tool] = None,
    format: Union[str, dict] = None,
    options: OllamaChatOptions = None,
    stream: bool = True,
    keep_alive: Union[str, int] = "15m",
    template: str = None,
    track: Track = None,
) -> Union[ChatResponseInfo, Generator[str, None, None]]:
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

    Returns:
        Union[dict, Generator[str, None, None]]: Either the JSON response or a generator for streamed responses.
    """
    from aim import Run, Text

    URL = "http://localhost:11434/api/chat"

    if isinstance(messages, str):
        messages = [
            Message(content=messages, role=MessageRole.USER)
        ]

    char_count = len(str(messages))
    token_count = token_counter(messages, model=model)

    logger.newline()
    logger.log("Model:", model, colors=["GRAY", "INFO"])
    logger.log("Prompt:", char_count, colors=["GRAY", "INFO"])
    logger.log("Tokens:", token_count, colors=["GRAY", "INFO"])
    logger.debug("Generating response...")

    if not any(message['role'] == MessageRole.USER for message in messages):
        messages[-1]["role"] = MessageRole.USER

    if system and not any(message['role'] == MessageRole.SYSTEM for message in messages):
        messages.insert(0, Message(
            content=system, role=MessageRole.SYSTEM))

    if tools:
        stream = False

    options = {**DEFAULT_SETTINGS, **(options or {})}

    # Format tool outputs to string if any
    messages = convert_tool_outputs_to_string(messages)

    # Prepare the request body
    body = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "keep_alive": keep_alive,
        "template": template,
        # "raw": False,
        "tools": tools,
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
    headers = {
        "Tokens": str(token_count),  # Include the token count here
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
                                response_info: ChatResponseInfo = decoded_chunk.copy()
                                if not content:
                                    response_info["message"]["content"] = output
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

                            yield content

                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to decode JSON: {decoded_line}")

            return line_generator()
        else:
            response = r.json()
            response["options"] = options

            response_info: ChatResponseInfo = response.copy()
            output = response_info["message"]["content"]

            logger.success(output, flush=True)

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
    prompt = "Write a 20 word creative story about an explorer finding a hidden treasure."

    # No stream
    logger.newline()
    logger.info("No stream response:")
    response = call_ollama_chat(prompt, stream=False)
    logger.success(json.dumps(response, indent=2))

    # With stream
    logger.newline()
    logger.info("With stream response:")
    response = ""
    for chunk in call_ollama_chat(prompt):
        response += chunk
