import json
import requests
from enum import Enum
from typing import Generator, Literal, Optional, TypedDict, Union
from langchain_ollama.llms import OllamaLLM
from langchain_core.outputs.llm_result import LLMResult, GenerationChunk
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from jet.llm.llm_types import OllamaChatMessage, MessageRole, Track
from jet.logger import logger
from jet.transformers import make_serializable


# class CustomStreamingHandler(StreamingStdOutCallbackHandler):
#     """Custom callback handler to log each new token with `logger.success`."""

#     def on_llm_new_token(self, token: str, **kwargs: dict) -> None:
#         logger.success(token, end="", flush=True)


class Ollama:
    DEFAULT_SETTINGS = {
        "seed": 42,
        "temperature": 1,
        "top_k": 40,
        "top_p": 0.5,
        "tfs_z": 1.9,
        "stop": [],
        "num_keep": 1,
        "num_predict": -2,
    }

    def __init__(self, model: str = "mistral", base_url: str = "http://jetairm1:11434"):
        self.model = model
        self.base_url = base_url
        self.ollama = OllamaLLM(model=model, base_url=base_url)

    def generate(self, prompt: str, settings: dict[str, any] = None, raw: bool = False) -> dict[str, any]:
        # Merge default settings with user-provided settings
        settings = {**self.DEFAULT_SETTINGS, **(settings or {})}

        data: LLMResult = self.ollama.generate(
            prompts=[prompt],
            options={"stream": True, "raw": raw, **settings},
            # callbacks=[CustomStreamingHandler()],
        )
        generated_chunk: GenerationChunk = data.generations[0][0]
        output = generated_chunk.text.strip()

        return {
            "prompt": prompt,
            "output": output,
            "meta": {
                "prompt_len": len(prompt),
                "output_len": len(output),
                "total_len": len(prompt) + len(output),
            },
            "settings": settings,
        }


def call_ollama_chat(
    messages: str | list[OllamaChatMessage],
    model: str = "llama3.1",
    *,
    system: str = "",
    tools: list = None,
    format: Union[str, dict] = None,
    options: dict = None,
    stream: bool = True,
    track: Track = None,
) -> Union[dict, Generator[str, None, None]]:
    """
    Wraps call_ollama_chat to track the prompt and response using Aim.

    Args:
        model (str | list[OllamaChatMessage]): The name of the model to use.
        messages (list): The messages for the conversation.
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
            OllamaChatMessage(content=messages, role=MessageRole.USER)
        ]

    if not any(message['role'] == MessageRole.SYSTEM for message in messages):
        messages.insert(0, OllamaChatMessage(
            content=system, role=MessageRole.SYSTEM))

    if tools:
        stream = False

    # Default options with provided overrides
    default_options = {
        "stream": stream,
        "num_ctx": 4096,
        "temperature": 0.7,
    }
    options = {**default_options, **(options or {})}

    # Prepare the request body
    body = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "format": format,
        "options": options,
    }
    body = make_serializable(body)

    # Initialize Aim run
    if track:
        track_settings: Track = {
            "log_system_params": True,
            **track.copy()
        }
        del track_settings["run_name"]
        del track_settings["metadata"]
        logger.log("Track Settings:", json.dumps(
            track_settings, indent=2), ["GRAY", "INFO"])
        run = Run(**track_settings)

    try:
        # Make the POST request
        r = requests.post(
            url=URL,
            json=body,
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
                            decoded_chunk = json.loads(decoded_line)
                            content = decoded_chunk.get(
                                "message", {}).get("content", "")
                            response_chunks.append(content)

                            if decoded_chunk.get("done"):
                                logger.debug("Stream completed.")
                                # For Aim tracking
                                if track:
                                    # Log the prompt (messages) to Aim
                                    prompt = messages[-1]["content"]
                                    response = "".join(response_chunks)

                                    value = {
                                        'prompt': prompt,
                                        'response': response,
                                    }
                                    aim_text = Text(
                                        json.dumps(value, indent=2))
                                    run.track(
                                        aim_text,
                                        name=track['run_name'],
                                        context={
                                            "model": model,
                                            "options": options,
                                            "metadata": track['metadata'],
                                        }
                                    )

                            yield content

                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to decode JSON: {decoded_line}")

            return line_generator()
        else:
            response = r.json()

            return response

    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        if track:
            run.track({"error": str(e)}, name="error",
                      context={"model": model})
        return {"error": str(e)}

    except Exception as e:
        logger.error("Unexpected error")
        return {"error": str(e)}

    finally:
        if track:
            run.close()  # Ensure run is closed in all cases


# Main function to demonstrate sample usage
if __name__ == "__main__":
    prompt = "Write a creative story about an explorer finding a hidden treasure."

    # generator = Ollama()
    # result = generator.generate(prompt)
    # print("Generated Output:")
    # print(result["output"])

    response = ""
    for chunk in call_ollama_chat(prompt):
        response += chunk
        logger.success(chunk, flush=True)
