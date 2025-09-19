from loguru import logger
import subprocess
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Union, Dict, Any, Callable
from jet.llm.mlx.logger_utils import ChatLogger
from jet.llm.mlx.config import DEFAULT_OLLAMA_LOG_DIR
from jet.transformers.formatters import format_json
from jet.logger import logger as jet_logger
from swarms.agents.openai_assistant import OpenAIAssistant
from swarms.structs.agent import Agent
import json

from jet.utils.text import format_sub_dir

try:
    import ollama
except ImportError:
    logger.error("Failed to import ollama")
    subprocess.run(["pip", "install", "ollama"])
    import ollama

class Message(BaseModel):
    role: str = Field(
        ...,
        pattern="^(user|system|assistant)$",
        description="The role of the message sender.",
    )
    content: str = Field(
        ..., min_length=1, description="The content of the message."
    )

class OllamaFunctionCaller(OpenAIAssistant):
    def __init__(
        self,
        model_name: str = "llama3.2",
        host: Optional[str] = None,
        timeout: int = 30,
        stream: bool = False,
        temperature: float = 0.1,
        agent_name: Optional[str] = None,
        name: str = "Ollama Assistant",
        description: str = "Ollama-based assistant wrapper",
        instructions: Optional[str] = None,
        tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
        file_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ):
        """
        Initializes the OllamaFunctionCaller with the model name and optional parameters.
        Args:
            model_name (str): The name of the model to interact with (e.g., 'llama3.2').
            host (str, optional): The Ollama host to connect to. Defaults to None.
            timeout (int, optional): Timeout for the requests. Defaults to 30 seconds.
            stream (bool, optional): Enable streaming for responses. Defaults to False.
            temperature (float, optional): Temperature for text generation. Defaults to 0.1.
            agent_name (str, optional): Name of the agent for logging purposes.
            name (str): Name of the assistant (inherited from OpenAIAssistant).
            description (str): Description of the assistant.
            instructions (str, optional): System instructions for the assistant.
            tools (List[Union[Dict[str, Any], Callable]], optional): List of tools, either as dictionaries or callable functions.
            file_ids (List[str], optional): List of file IDs to attach.
            metadata (Dict[str, Any], optional): Additional metadata.
            functions (List[Dict[str, Any]], optional): List of custom functions in dictionary format.
        """
        # Initialize attributes before super().__init__ to avoid OpenAI client setup
        self.name = name
        self.description = description
        self.instructions = instructions
        self.model = model_name
        self.tools = []
        self.file_ids = file_ids
        self.metadata = metadata or {}
        self.functions = functions
        self.available_functions: Dict[str, Callable] = {}

        # Debug logs to inspect initialization
        jet_logger.debug(f"Initializing OllamaFunctionCaller with model: {model_name}")
        jet_logger.debug(f"Tools: {tools}, Functions: {functions}")

        # Process tools (dictionaries or callables)
        if tools:
            for tool in tools:
                if isinstance(tool, dict):
                    self.tools.append(tool)
                    if tool.get("type") == "function" and "name" in tool.get("function", {}):
                        jet_logger.debug(f"Added dictionary-based tool: {tool['function']['name']}")
                elif callable(tool):
                    func_dict = {
                        "type": "function",
                        "function": {
                            "name": tool.__name__,
                            "description": tool.__doc__ or f"Function {tool.__name__}",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                        }
                    }
                    self.tools.append(func_dict)
                    self.available_functions[tool.__name__] = tool
                    jet_logger.debug(f"Added callable tool: {tool.__name__}")
                else:
                    jet_logger.error(f"Invalid tool type: {type(tool)}")

        # Handle functions parameter
        if functions:
            for func in functions:
                self.tools.append({"type": "function", "function": func})
                if "name" in func:
                    jet_logger.debug(f"Added function from functions param: {func['name']}")

        # Call parent __init__ without triggering OpenAI client
        super(Agent, self).__init__(*args, **kwargs)  # Skip OpenAIAssistant.__init__

        # Initialize Ollama-specific attributes
        self.model_name = model_name
        self.host = host
        self.timeout = timeout
        self.stream = stream
        self.temperature = temperature
        self.agent_name = agent_name
        self.client = ollama.Client(host=host) if host else ollama.Client()
        self.messages: List[Dict[str, str]] = []
        if instructions:
            self.messages.append({"role": "system", "content": instructions})
        log_dir = DEFAULT_OLLAMA_LOG_DIR
        if agent_name:
            log_dir += f"/{format_sub_dir(agent_name)}"
        self._chat_logger = ChatLogger(log_dir)

        # Override OpenAI-specific attributes
        self.client = self  # Use self as client to redirect to Ollama methods
        self.assistant = self  # Use self as assistant to avoid OpenAI API calls

        jet_logger.debug("OllamaFunctionCaller initialization completed")
    def add_function(
        self,
        func: Callable,
        description: str,
        parameters: Dict[str, Any],
    ) -> None:
        """
        Add a function that the assistant can simulate calling.
        Args:
            func: The function to make available to the assistant.
            description: Description of what the function does.
            parameters: JSON schema describing the function parameters.
        """
        func_dict = {
            "name": func.__name__,
            "description": description,
            "parameters": parameters,
        }
        self.tools.append({"type": "function", "function": func_dict})
        self.available_functions[func.__name__] = func

    def add_message(
        self, content: str, file_ids: Optional[List[str]] = None
    ) -> None:
        """
        Add a message to the internal messages list for Ollama.
        Args:
            content (str): The text content of the message to add.
            file_ids (List[str], optional): Ignored for Ollama compatibility.
        """
        self.messages.append({"role": "user", "content": content})

    def _get_response(self) -> str:
        """
        Get the latest assistant response using Ollama's chat method.
        Returns:
            str: The assistant's response as a string.
        """
        # Debug log to inspect messages before validation
        jet_logger.debug(f"Messages before validation: {self.messages}")

        validated_messages = self.validate_messages(
            [Message(**msg) for msg in self.messages]
        )

        # Debug log to inspect validated messages
        jet_logger.debug(f"Validated messages: {validated_messages}")

        if not validated_messages:
            jet_logger.error("No validated messages available")
            return ""

        # Check for function calls in the last user message
        last_message = validated_messages[-1] if validated_messages else None
        if last_message and last_message["role"] == "user":
            try:
                # Simulate function call detection by checking for a JSON-like structure
                content = json.loads(last_message["content"])
                jet_logger.debug(f"Parsed content for function call: {content}")
                if isinstance(content, dict) and "function" in content:
                    function_name = content["function"]
                    function_args = content.get("arguments", {})
                    if function_name in self.available_functions:
                        function_response = self.available_functions[function_name](**function_args)
                        jet_logger.debug(f"Function {function_name} response: {function_response}")
                        self.messages.append({"role": "assistant", "content": str(function_response)})
                        return str(function_response)
            except json.JSONDecodeError:
                jet_logger.debug("No function call detected, proceeding with chat")

        options = {"temperature": self.temperature}
        response_text = ""
        try:
            stream = self.chat(
                messages=[Message(**msg) for msg in self.messages],
                options=options,
            )
            jet_logger.debug(f"Chat stream response: {stream}")
            if stream:
                response_text = stream
                self.messages.append({"role": "assistant", "content": response_text})
            return response_text
        except Exception as e:
            jet_logger.error(f"Error getting response: {e}")
            return ""

    def run(self, task: str, *args, **kwargs) -> str:
        """
        Run a task using the Ollama model, compatible with OpenAIAssistant interface.
        Args:
            task (str): The task or prompt to send to the assistant.
        Returns:
            str: The assistant's response as a string.
        """
        self.add_message(task)
        return self._get_response()

    def batch_run(
        self, tasks: List[str], *args, **kwargs
    ) -> List[Any]:
        """
        Run a batch of tasks sequentially using the Ollama model.
        Args:
            tasks (List[str]): List of tasks to execute.
        Returns:
            List[Any]: List of responses from the assistant.
        """
        results = []
        for task in tasks:
            self.messages = []  # Reset messages for each task
            if self.instructions:
                self.messages.append({"role": "system", "content": self.instructions})
            results.append(self.run(task, *args, **kwargs))
        return results

    def run_concurrently(
        self, tasks: List[str], *args, **kwargs
    ) -> List[Any]:
        """
        Run a batch of tasks concurrently using the Ollama model.
        Note: Ollama's client may not support true concurrency, so this simulates it.
        Args:
            tasks (List[str]): List of tasks to execute.
        Returns:
            List[Any]: List of responses from the assistant.
        """
        from concurrent.futures import ThreadPoolExecutor
        import os

        def run_task(task: str) -> str:
            # Create a new instance for thread safety
            instance = OllamaFunctionCaller(
                model_name=self.model_name,
                host=self.host,
                timeout=self.timeout,
                stream=self.stream,
                temperature=self.temperature,
                agent_name=self.agent_name,
                name=self.name,
                description=self.description,
                instructions=self.instructions,
                tools=self.tools,
                file_ids=self.file_ids,
                metadata=self.metadata,
            )
            return instance.run(task, *args, **kwargs)

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            return list(executor.map(run_task, tasks))

    def _ensure_thread(self):
        """
        Override to maintain compatibility; Ollama does not use threads.
        Messages are stored internally in self.messages.
        """
        pass

    def _wait_for_run(self, run) -> Any:
        """
        Override to bypass OpenAI run polling; not applicable for Ollama.
        """
        return self

    def _handle_tool_calls(self, run, thread_id: str) -> None:
        """
        Override to bypass OpenAI tool calls; handled in _get_response for Ollama.
        """
        pass

    def chat(
        self, messages: List[Message], *args, **kwargs
    ) -> Union[str, None]:
        """Executes the chat task with the Ollama model."""
        # Debug log to inspect input messages
        jet_logger.debug(f"Chat method called with messages: {messages}")

        validated_messages = self.validate_messages(messages)

        # Debug log to inspect validated messages
        jet_logger.debug(f"Validated messages in chat: {validated_messages}")

        if not validated_messages:
            jet_logger.error("No validated messages available for chat")
            return None

        options = {"temperature": self.temperature, **kwargs.pop("options", {}), **kwargs}

        # Debug log to inspect chat settings
        jet_logger.gray("OllamaFunctionCaller Chat Settings:")
        jet_logger.debug(format_json({
            "messages": validated_messages,
            "model": self.model_name,
            "options": options,
        }))

        response_text = ""
        try:
            stream = ollama.chat(
                model=self.model_name,
                messages=validated_messages,
                stream=True,
                options=options,
                *args,
                **kwargs,
            )
            for chunk in stream:
                content = chunk["message"]["content"]
                jet_logger.teal(content, flush=True)
                response_text += content

            # Debug log to inspect final response
            jet_logger.debug(f"Chat response: {response_text}")

            self._chat_logger.log_interaction(
                messages=validated_messages,
                response={"content": response_text},
                model=self.model_name,
                options=options,
                method="stream_chat",
            )
            return response_text
        except Exception as e:
            jet_logger.error(f"Error in chat method: {e}")
            return None

    def generate(self, prompt: str, *args, **kwargs) -> Optional[str]:
        """Generates text based on a prompt."""
        # Debug log to inspect input prompt
        jet_logger.debug(f"Generate method called with prompt: {prompt}")

        if len(prompt) == 0:
            jet_logger.error("Prompt cannot be empty.")
            return None

        options = {"temperature": self.temperature, **kwargs.pop("options", {}), **kwargs}

        # Debug log to inspect generate settings
        jet_logger.gray("OllamaFunctionCaller Generate Settings:")
        jet_logger.debug(format_json({
            "prompt": prompt,
            "model": self.model_name,
            "options": options,
        }))

        response_text = ""
        try:
            stream = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=True,
                options=options,
                *args,
                **kwargs,
            )
            for chunk in stream:
                content = chunk.get("response", "")
                jet_logger.teal(content, flush=True)
                response_text += content

            # Debug log to inspect final response
            jet_logger.debug(f"Generate response: {response_text}")

            self._chat_logger.log_interaction(
                messages=[{"role": "user", "content": prompt}],
                response={"content": response_text},
                model=self.model_name,
                options=options,
                method="stream_generate",
            )
            return response_text
        except Exception as e:
            jet_logger.error(f"Error in generate method: {e}")
            return None

    def validate_messages(self, messages: List[Message]) -> List[dict]:
        """
        Validates the list of messages using Pydantic schema.
        Args:
            messages (List[Message]): List of messages to validate.
        Returns:
            List[dict]: Validated messages in dictionary format.
        """
        try:
            return [message.dict() for message in messages]
        except ValidationError as e:
            jet_logger.error(f"Validation error: {e}")
            return []

    def list_models(self) -> List[str]:
        """
        Lists available models in the Ollama environment.
        Returns:
            List[str]: List of available model names.
        """
        # Debug log to inspect method call
        jet_logger.debug("list_models method called")

        try:
            models = ollama.list()
            model_names = [m.model for m in models.models]

            # Debug log to inspect retrieved models
            jet_logger.debug(f"Retrieved models: {model_names}")

            self._chat_logger.log_interaction(
                messages=[],
                response={"models": model_names},
                model=self.model_name,
                options={},
                method="list_models",
            )
            return model_names
        except Exception as e:
            jet_logger.error(f"Error listing models: {e}")
            return []

    def show_model(self) -> Dict[str, Any]:
        """
        Shows details of the current model.
        Returns:
            Dict[str, Any]: Details of the current model.
        """
        # Debug log to inspect method call
        jet_logger.debug(f"show_model method called for model: {self.model_name}")

        try:
            model_details = ollama.show(self.model_name)

            # Debug log to inspect model details
            jet_logger.debug(f"Model details: {model_details}")

            self._chat_logger.log_interaction(
                messages=[],
                response={"model_details": model_details},
                model=self.model_name,
                options={},
                method="show_model",
            )
            return model_details
        except Exception as e:
            jet_logger.error(f"Error showing model details: {e}")
            return {}
