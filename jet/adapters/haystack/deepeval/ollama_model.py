from ollama import Client, AsyncClient, ChatResponse
from typing import AsyncIterator, Iterator, Optional, Tuple, Union, Dict
from pydantic import BaseModel

from deepeval.models.retry_policy import (
    create_retry_decorator,
)

from deepeval.models import DeepEvalBaseLLM
from deepeval.key_handler import ModelKeyValues, KEY_FILE_HANDLER
from deepeval.constants import ProviderSlug as PS

import os
from jet.llm.mlx.config import DEFAULT_OLLAMA_LOG_DIR
from jet.llm.logger_utils import ChatLogger
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.text import format_sub_dir


retry_ollama = create_retry_decorator(PS.OLLAMA)


class OllamaModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0,
        generation_kwargs: Optional[Dict] = None,
        log_dir: Optional[str] = None,
        agent_name: Optional[str] = None,
        verbose: bool = True,
        **kwargs,
    ):
        model_name = model or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.LOCAL_MODEL_NAME
        )
        self.base_url = (
            base_url
            or KEY_FILE_HANDLER.fetch_data(ModelKeyValues.LOCAL_MODEL_BASE_URL)
            or "http://localhost:11434"
        )
        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model_name)

        log_dir = os.path.join(DEFAULT_OLLAMA_LOG_DIR, log_dir or "")
        if agent_name:
            log_dir += f"/{format_sub_dir(agent_name)}"
        self._chat_logger = ChatLogger(log_dir, method="chat")
        self.verbose = verbose

    ###############################################
    # Other generate functions
    ###############################################

    @retry_ollama
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        chat_model: Client = self.load_model()
        if self.verbose:
            logger.gray("Ollama Generator Settings:")
            logger.info(format_json({
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "options": {
                    **{"temperature": self.temperature},
                    **self.generation_kwargs,
                },
                "format": schema.model_json_schema() if schema else None,
            }))
        response_stream: Iterator[ChatResponse] = chat_model.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            format=schema.model_json_schema() if schema else None,
            options={
                **{"temperature": self.temperature},
                **self.generation_kwargs,
            },
            stream=True,
        )
        content = ""
        for chunk in response_stream:
            if self.verbose:
                logger.teal(chunk.message.content, flush=True)
            content += chunk.message.content
        
        self._chat_logger.log_interaction(
            messages=[{"role": "user", "content": prompt}],
            response=content,
            model=self.model_name,
            format=schema.model_json_schema() if schema else None,
            options={
                **{"temperature": self.temperature},
                **self.generation_kwargs,
            },
            response_meta=chunk
        )

        return (
            (
                schema.model_validate_json(content)
                if schema
                else content
            ),
            0,
        )

    @retry_ollama
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[str, float]:
        chat_model: AsyncClient = self.load_model(async_mode=True)
        if self.verbose:
            logger.gray("Ollama Async Generator Settings:")
            logger.info(format_json({
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "options": {
                    **{"temperature": self.temperature},
                    **self.generation_kwargs,
                },
                "format": schema.model_json_schema() if schema else None,
            }))
        response_stream: AsyncIterator[ChatResponse] = await chat_model.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            format=schema.model_json_schema() if schema else None,
            options={
                **{"temperature": self.temperature},
                **self.generation_kwargs,
            },
            stream=True,
        )
        content = ""
        async for chunk in response_stream:
            if self.verbose:
                logger.teal(chunk.message.content, flush=True)
            content += chunk.message.content

        self._chat_logger.log_interaction(
            messages=[{"role": "user", "content": prompt}],
            response=content,
            model=self.model_name,
            format=schema.model_json_schema() if schema else None,
            options={
                **{"temperature": self.temperature},
                **self.generation_kwargs,
            },
            response_meta=chunk
        )

        return (
            (
                schema.model_validate_json(content)
                if schema
                else content
            ),
            0,
        )

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return self._build_client(Client)
        return self._build_client(AsyncClient)

    def _build_client(self, cls):
        return cls(host=self.base_url, **self.kwargs)

    def get_model_name(self):
        return f"{self.model_name} (Ollama)"
