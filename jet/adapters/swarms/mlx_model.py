from loguru import logger
import subprocess
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Union

try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
except ImportError:
    logger.error("Failed to import mlx_lm")
    subprocess.run(["pip", "install", "mlx-lm"])
    from mlx_lm import load, generate


class Message(BaseModel):
    role: str = Field(
        ...,
        pattern="^(user|system|assistant)$",
        description="The role of the message sender.",
    )
    content: str = Field(..., min_length=1, description="The content of the message.")


class MLXModel:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        stream: bool = False,
    ):
        """
        Initializes the MLXModel with the model name and optional parameters.

        Args:
            model_name (str): The name of the MLX model to load.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 500.
            stream (bool, optional): Enable streaming output. Defaults to False.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream

        self.model, self.tokenizer = load(model_name)

    def validate_messages(self, messages: List[Message]) -> List[Message]:
        """Validates the list of messages using Pydantic schema."""
        try:
            return [Message(**msg.dict()) for msg in messages]
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            return []

    def chat(self, messages: List[Message], *args, **kwargs) -> Union[str, None]:
        """Executes the chat task by constructing a prompt."""
        validated_messages = self.validate_messages(messages)
        if not validated_messages:
            return None

        # Build a chat-style prompt
        prompt = ""
        for m in validated_messages:
            prompt += f"<|{m.role}|>\n{m.content}\n"
        prompt += "<|assistant|>\n"

        sampler = make_sampler(
            temp=self.temperature,
        )

        if self.stream:
            # Streaming: yield token-by-token
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                stream=True,
                sampler=sampler,
                *args,
                **kwargs,
            )
            for token in response:
                print(token, end="", flush=True)
            return None
        else:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                temp=self.temperature,
                max_tokens=self.max_tokens,
                *args,
                **kwargs,
            )
            return response

    def generate(self, prompt: str, *args, **kwargs) -> Optional[str]:
        """Generates text based on a prompt."""
        if len(prompt.strip()) == 0:
            logger.error("Prompt cannot be empty.")
            return None

        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            temp=self.temperature,
            max_tokens=self.max_tokens,
            *args,
            **kwargs,
        )

    def list_models(self) -> List[str]:
        """Lists available models (stub, since MLX does not have a registry)."""
        logger.warning(
            "list_models is not supported in MLX. Returning [self.model_name].")
        return [self.model_name]

    def show_model(self) -> dict:
        """Shows details of the current model (stub)."""
        return {"model_name": self.model_name, "temperature": self.temperature, "max_tokens": self.max_tokens}

    def create_model(self, modelfile: str) -> dict:
        """Not supported in MLX."""
        logger.warning("create_model is not supported in MLX.")
        return {}

    def delete_model(self) -> bool:
        """Not supported in MLX."""
        logger.warning("delete_model is not supported in MLX.")
        return False

    def run(self, task: str, *args, **kwargs):
        """Shortcut to generate output for a single task string."""
        return self.generate(task, *args, **kwargs)
