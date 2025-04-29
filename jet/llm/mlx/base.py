from typing import Callable
import json
import os
from mlx_lm import generate, load
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache


def mlx_chat(
    model_path: str,
    messages: list[dict],
    cache_path: str | None = None,
    verbose: bool = True,
) -> str:
    """
    Perform a multi-turn chat using mlx_lm with prompt caching.

    Args:
        model_path (str): Path or identifier for the model to load.
        messages (list[dict]): List of messages (role: user/assistant, content: str).
        cache_path (str, optional): Path to save/load the prompt cache.
        verbose (bool, optional): Enable verbose generation logs.

    Returns:
        str: The generated assistant response.
    """
    model, tokenizer = load(model_path)

    if cache_path and os.path.exists(cache_path):
        prompt_cache = load_prompt_cache(cache_path)
    else:
        prompt_cache = make_prompt_cache(model)

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True)

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        verbose=verbose,
        prompt_cache=prompt_cache,
    )

    if cache_path:
        save_prompt_cache(cache_path, prompt_cache)

    return response


def mlx_chat_with_tools(
    model_path: str,
    messages: list[dict],
    tools: dict[str, Callable],
    cache_path: str | None = None,
    verbose: bool = True,
) -> str:
    """
    Perform a chat with tool usage using mlx_lm and prompt caching.

    Args:
        model_path (str): Path or identifier for the model to load.
        messages (list[dict]): List of messages (role: user/assistant/tool, content: str).
        tools (dict[str, Callable]): Dictionary of available tool functions.
        cache_path (str, optional): Path to save/load the prompt cache.
        verbose (bool, optional): Enable verbose generation logs.

    Returns:
        str: The final assistant response after tool execution.
    """
    model, tokenizer = load(model_path)

    if cache_path and os.path.exists(cache_path):
        prompt_cache = load_prompt_cache(cache_path)
    else:
        prompt_cache = make_prompt_cache(model)

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tools=list(tools.values())
    )

    # First pass: detect tool call
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=2048,
        verbose=verbose,
        prompt_cache=prompt_cache,
    )

    # Extract tool call (model-specific parsing)
    tool_open, tool_close = "<tool_call>", "</tool_call>"
    start_tool = response.find(tool_open) + len(tool_open)
    end_tool = response.find(tool_close)

    if start_tool == -1 or end_tool == -1:
        raise ValueError("No tool call detected in the model's response.")

    tool_call = json.loads(response[start_tool:end_tool].strip())
    tool_result = tools[tool_call["name"]](**tool_call["arguments"])

    # Second pass: respond after tool result
    tool_message = [{
        "role": "tool",
        "name": tool_call["name"],
        "content": tool_result
    }]
    prompt = tokenizer.apply_chat_template(
        tool_message,
        add_generation_prompt=True
    )

    final_response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=2048,
        verbose=verbose,
        prompt_cache=prompt_cache,
    )

    if cache_path:
        save_prompt_cache(cache_path, prompt_cache)

    return final_response


class MLX:
    def __init__(self, model_path="mlx-community/Llama-3.2-3B-Instruct-4bit", cache_file=None):
        """
        Initialize the ChatModel with a specified model and optional prompt cache.

        Args:
            model_path (str): Path or Hugging Face repo for the model.
            cache_file (str, optional): Path to a prompt cache file to load.
        """
        self.model, self.tokenizer = load(model_path)
        self.prompt_cache = make_prompt_cache(
            self.model) if cache_file is None else load_prompt_cache(cache_file)
        self.conversation = []

    def chat(self, user_input: str | list, max_tokens=1000, verbose=True):
        """
        Process a user input and generate a response, maintaining conversation history.

        Args:
            user_input (str): The user's input message.
            max_tokens (int): Maximum number of tokens to generate.
            verbose (bool): Whether to print verbose output.

        Returns:
            str: The generated response.
        """
        # Add user input to conversation history
        if isinstance(user_input, str):
            self.conversation.append({"role": "user", "content": user_input})
        else:
            self.conversation = user_input

        # Generate prompt with chat template
        prompt = self.tokenizer.apply_chat_template(
            conversation=self.conversation,
            add_generation_prompt=True
        )

        # Generate response
        response = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=verbose,
            prompt_cache=self.prompt_cache
        )

        # Add assistant response to conversation history
        self.conversation.append({"role": "assistant", "content": response})

        return response

    def generate(self, prompt, max_tokens=1000, verbose=True):
        """
        Generate a response for a single prompt without maintaining conversation history.

        Args:
            prompt (str): The input prompt.
            max_tokens (int): Maximum number of tokens to generate.
            verbose (bool): Whether to print verbose output.

        Returns:
            str: The generated response.
        """
        # Format prompt as a single user message
        conversation = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=True
        )

        # Generate response
        response = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            verbose=verbose,
            prompt_cache=self.prompt_cache
        )

        return response

    def save_cache(self, cache_path):
        """
        Save the prompt cache to a file.

        Args:
            cache_path (str): Path to save the prompt cache file.
        """
        save_prompt_cache(cache_path, self.prompt_cache)

    def reset_conversation(self):
        """
        Reset the conversation history.
        """
        self.conversation = []
