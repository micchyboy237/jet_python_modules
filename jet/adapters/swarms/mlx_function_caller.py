import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Type, Any, Iterator, Union
from pydantic import BaseModel
from jet.llm.mlx.remote import generation as gen
from jet.llm.mlx.remote.types import Message, ChatCompletionResponse
from jet.logger import logger

SUPPORTED_MLX_MODELS = [
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
]

class MLXFunctionCaller:
    """
    A class to interact with MLX remote models for generating text or structured data, with streaming and tool calling support.
    Compatible with swarms Agent without litellm dependency.
    """

    supports_function_calling = True  # Explicitly indicate function calling support
    model_name: str  # Ensure model_name is typed

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        base_model: Optional[Type[BaseModel]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        tools: Optional[List[Callable]] = None,
        stream: bool = False,
        base_url: Optional[str] = None,
        verbose: bool = False
    ):
        if not model_name:
            raise ValueError("model_name must be provided")
        if model_name not in SUPPORTED_MLX_MODELS:
            logger.warning(f"Model {model_name} not in {SUPPORTED_MLX_MODELS}. Ensure MLX server supports it.")
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.base_model = base_model
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.tools = tools
        self.stream = stream
        self.base_url = base_url or "http://localhost:8080"  # Default MLX server URL
        self.verbose = verbose

    def __call__(self, task: str, *args, **kwargs) -> Union[str, Iterator[str], Any]:
        """Make MLXFunctionCaller callable to match Agent's expectations."""
        return self.run(task, *args, **kwargs)

    def run(self, task: str, *args, **kwargs) -> Union[str, Iterator[str], Any]:
        """
        Run the LLM with the given task, supporting streaming, structured output, and tool calling.
        Returns a string (non-streaming), iterator of strings (streaming), or BaseModel instance (structured output).
        """
        try:
            messages: List[Message] = []
            system_message = self.system_prompt or ""
            if self.base_model:
                system_message += (
                    f"\nReturn the response as a JSON object containing only the data fields defined in the following schema, "
                    f"without including the schema itself or any additional metadata:\n"
                    f"{self.base_model.model_json_schema()}\n"
                    f"For example, if the schema defines fields 'name' and 'age', return only {{\"name\": \"value\", \"age\": number}}."
                )
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": task})

            if self.stream:
                chunks = gen.stream_chat(
                    messages=messages,
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    tools=self.tools,
                    base_url=self.base_url,
                    verbose=self.verbose,
                    response_format="json" if self.base_model else "text"
                )
                for chunk in chunks:
                    content = self._extract_chunk_content(chunk)
                    if content:
                        yield content
                    if chunk.get("tool_calls"):
                        yield from self._handle_tool_calls(chunk)
            else:
                response = gen.chat(
                    messages=messages,
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    tools=self.tools,
                    base_url=self.base_url,
                    verbose=self.verbose,
                    response_format="json" if self.base_model else "text"
                )
                content = response.get("content", "")
                if response.get("tool_calls"):
                    content += "".join(self._handle_tool_calls(response))
                if self.base_model:
                    return self._parse_structured_response(content)
                return content

        except Exception as e:
            logger.error(f"Error in MLXFunctionCaller.run: {e}")
            return None

    def _extract_chunk_content(self, chunk: ChatCompletionResponse) -> Optional[str]:
        """Extract content from a streaming chunk, handling MLX response format."""
        try:
            for choice in chunk.get("choices", []):
                message = choice.get("message", {})
                content = message.get("content")
                if content:
                    return content
                delta = choice.get("delta", {})
                if delta.get("content"):
                    return delta["content"]
            return None
        except Exception as e:
            logger.warning(f"Failed to extract chunk content: {e}")
            return None

    def _handle_tool_calls(self, response: ChatCompletionResponse) -> Iterator[str]:
        """Handle tool calls in the response, yielding results as strings."""
        try:
            tool_calls = response.get("tool_calls", [])
            if not tool_calls:
                return
            for call in tool_calls:
                if call.get("type") == "function":
                    func_name = call["function"]["name"]
                    args = call["function"]["arguments"]
                    if isinstance(args, dict):
                        args_str = json.dumps(args)
                    else:
                        args = json.loads(args) if isinstance(args, str) else {}
                        args_str = json.dumps(args)
                    yield f'[TOOL_CALL] {func_name}({args_str})'
        except Exception as e:
            logger.warning(f"Error handling tool calls: {e}")
            return

    def _parse_structured_response(self, response_text: str) -> Optional[Any]:
        """Parse response text into a BaseModel instance if base_model is specified."""
        if not self.base_model:
            return response_text
        try:
            cleaned_response = re.sub(r'^```json\n|```$', '', response_text.strip(), flags=re.MULTILINE)
            json_data = json.loads(cleaned_response)
            if isinstance(json_data, dict) and 'properties' in json_data:
                json_data = json_data['properties']
            return self.base_model.model_validate(json_data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing structured response: {e}")
            return None

    def check_model_support(self) -> List[str]:
        """List supported models."""
        for model in SUPPORTED_MLX_MODELS:
            print(model)
        return SUPPORTED_MLX_MODELS

    def batch_run(self, tasks: List[str]) -> List[Any]:
        """Run multiple tasks sequentially."""
        return [self.run(task) for task in tasks]

    def concurrent_run(self, tasks: List[str]) -> List[Any]:
        """Run multiple tasks concurrently."""
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            return list(executor.map(self.run, tasks))