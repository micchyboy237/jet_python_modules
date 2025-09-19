from loguru import logger
import subprocess
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Union
from jet.llm.mlx.logger_utils import ChatLogger
from jet.llm.mlx.config import DEFAULT_OLLAMA_LOG_DIR
from jet.transformers.formatters import format_json
from jet.logger import logger as jet_logger
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

class OllamaModel:
    def __init__(
        self,
        model_name: str = "llama3.2",
        host: Optional[str] = None,
        timeout: int = 30,
        stream: bool = False,
        temperature: float = 0.1,
        agent_name: Optional[str] = None,
    ):
        """
        Initializes the OllamaModel with the model name and optional parameters.
        Args:
            model_name (str): The name of the model to interact with (e.g., 'llama3.1').
            host (str, optional): The Ollama host to connect to. Defaults to None.
            timeout (int, optional): Timeout for the requests. Defaults to 30 seconds.
            stream (bool, optional): Enable streaming for responses. Defaults to False.
            temperature (float, optional): Temperature for text generation. Defaults to 0.1.
        """
        if model_name.startswith("ollama/"):
            self.model_name = model_name[len("ollama/") :]
        else:
            self.model_name = model_name
        self.host = host
        self.timeout = timeout
        self.stream = stream
        self.temperature = temperature
        self.client = ollama.Client(host=host) if host else None
        self.agent_name = agent_name

        log_dir = DEFAULT_OLLAMA_LOG_DIR
        if agent_name:
            log_dir += f"/{format_sub_dir(agent_name)}"
        self._chat_logger = ChatLogger(log_dir)

    def validate_messages(
        self, messages: List[Message]
    ) -> List[dict]:
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
            print(f"Validation error: {e}")
            return []

    def chat(
        self, messages: List[Message], *args, **kwargs
    ) -> Union[str, None]:
        """Executes the chat task."""
        validated_messages = self.validate_messages(messages)
        if not validated_messages:
            return None

        options = {"temperature": self.temperature, **kwargs.pop("options", {}), **kwargs}
        jet_logger.gray("Ollama Model Chat Settings:")
        jet_logger.info(format_json({
            "messages": validated_messages,
            "model": self.model_name,
            "options": options,
        }))

        response_text = ""
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

        self._chat_logger.log_interaction(
            messages=validated_messages,
            response={"content": response_text},
            model=self.model_name,
            options=options,
            method="stream_chat",
        )
        return response_text

    def generate(self, prompt: str, *args, **kwargs) -> Optional[str]:
        """Generates text based on a prompt."""
        if len(prompt) == 0:
            jet_logger.error("Prompt cannot be empty.")
            return None

        options = {"temperature": self.temperature, **kwargs.pop("options", {}), **kwargs}
        jet_logger.gray("Ollama Model Generate Settings:")
        jet_logger.info(format_json({
            "prompt": prompt,
            "model": self.model_name,
            "options": options,
        }))

        response_text = ""
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

        self._chat_logger.log_interaction(
            messages=[{"role": "user", "content": prompt}],
            response={"content": response_text},
            model=self.model_name,
            options=options,
            method="stream_generate",
        )
        return response_text

    def list_models(self) -> List[str]:
        """Lists available models."""
        models = ollama.list()
        model_names = [m.model for m in models.models]
        return model_names

    def show_model(self) -> dict:
        """Shows details of the current model."""
        return ollama.show(self.model_name)

    def create_model(self, modelfile: str) -> dict:
        """Creates a new model from a modelfile."""
        return ollama.create(
            model=self.model_name, modelfile=modelfile
        )

    def delete_model(self) -> bool:
        """Deletes the current model."""
        try:
            ollama.delete(self.model_name)
            return True
        except ollama.ResponseError as e:
            print(f"Error deleting model: {e}")
            return False

    def run(self, task: str, *args, **kwargs):
        """
        Executes the task based on the task string.
        Args:
            task (str): The task to execute, such as 'chat', 'generate', etc.
        """
        return self.generate(prompt=task, *args, **kwargs)