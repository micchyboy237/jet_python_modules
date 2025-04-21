from typing import Iterator, Mapping, Optional, Dict, Any, Union, cast, override
from jet.actions.generation import call_ollama_chat
from jet.token.token_utils import token_counter
from jet.transformers.object import make_serializable
from jet.utils.class_utils import get_non_empty_attributes
from langchain_ollama import ChatOllama as BaseChatOllama, OllamaEmbeddings as BaseOllamaEmbeddings
from jet.logger import logger
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import ensure_config
from langchain_core.outputs import ChatGeneration
from jet.decorators.error import wrap_retry
from ollama import AsyncClient, Client, Message as OllamaMessage, Options

from ollama._types import ChatResponse
from shared.setup.events import EventSettings

DETERMINISTIC_LLM_SETTINGS = {
    "seed": 42,
    "temperature": 0,
    "num_keep": 0,
    "num_predict": -1,
}


class ChatOllama(BaseChatOllama):
    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        client_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        from jet.token.token_utils import token_counter

        # Generate headers dynamically
        event = EventSettings.call_ollama_chat_langchain()
        pre_start_hook_start_time = EventSettings.event_data["pre_start_hook"]["start_time"]
        log_filename = event["filename"].split(".")[0]
        logger.log("Log-Filename:", log_filename, colors=["WHITE", "DEBUG"])
        token_count = token_counter(kwargs.get(
            "messages", []), model=model) if kwargs.get("messages") else 0
        headers = {
            "Tokens": str(token_count),
            "Log-Filename": log_filename,
            "Event-Start-Time": pre_start_hook_start_time,
        }

        # Combine provided `client_kwargs` with default headers
        client_kwargs = client_kwargs or {}
        client_kwargs.setdefault("headers", headers)

        options = {**DETERMINISTIC_LLM_SETTINGS, **kwargs}

        # Call the parent class initializer with updated parameters
        super().__init__(model=model, base_url=base_url,
                         client_kwargs=client_kwargs, **options)

    def _chat_params(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from jet.llm.models import OLLAMA_MODEL_EMBEDDING_TOKENS, OLLAMA_MODEL_NAMES
        from jet.token.token_utils import token_counter

        max_token = 0.4

        params = super()._chat_params(
            messages,
            stop,
            **kwargs
        )
        options: Options = params["options"]
        model: OLLAMA_MODEL_NAMES = params["model"]
        messages = params["messages"]

        model_max_length = OLLAMA_MODEL_EMBEDDING_TOKENS[model]
        max_length = int(model_max_length * max_token)
        token_count = token_counter(messages, model)

        if token_count > model_max_length:
            error = f"token_count ({token_count}) must be less than model ({model}) max length ({model_max_length})"
            logger.warning(error)
            # raise ValueError(error)

        options.num_ctx = model_max_length
        return params

    def _create_chat_stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Iterator[Union[Mapping[str, Any], str]]:
        chat_params = self._chat_params(messages, stop, **kwargs)
        chat_params = make_serializable(chat_params)
        chat_params["options"] = get_non_empty_attributes(
            chat_params["options"])

        stream = not chat_params.get("tools") and chat_params["stream"]

        settings = {
            **chat_params,
            "full_stream_response": True,
        }

        response = call_ollama_chat(**settings)

        final_response = {}

        if not stream:
            content = response["message"]["content"]
            role = response["message"]["role"]
            tool_calls = response["message"].get("tool_calls", [])

            final_response_content = content
            final_response_tool_calls = tool_calls
            if final_response_tool_calls:
                final_response_content += f"\n{final_response_tool_calls}".strip()

            # prompt_token_count = token_counter(messages, model)
            # response_token_count = token_counter(
            #     final_response_content, model)

            final_response = {
                **response.copy(),
                # "usage": {
                #     "prompt_tokens": prompt_token_count,
                #     "completion_tokens": response_token_count,
                #     "total_tokens": prompt_token_count + response_token_count,
                # }
            }

        else:
            content = ""
            role = ""
            tool_calls = []

            if isinstance(response, dict) and "error" in response:
                raise ValueError(
                    f"Ollama API error:\n{response['error']}")

            for chunk in response:

                content += chunk["message"]["content"]
                if not role:
                    role = chunk["message"]["role"]
                if chunk["done"]:
                    # prompt_token_count: int = token_counter(
                    #     messages, model)
                    # response_token_count: int = token_counter(
                    #     content, model)

                    updated_chunk = chunk.copy()
                    updated_chunk["message"]["content"] = content

                    final_response = {
                        **updated_chunk,
                        # "usage": {
                        #     "prompt_tokens": prompt_token_count,
                        #     "completion_tokens": response_token_count,
                        #     "total_tokens": prompt_token_count + response_token_count,
                        # }
                    }

        final_response.pop("message")
        chat_response = ChatResponse(
            message=OllamaMessage(
                role=role,
                content=content,
                tool_calls=tool_calls,
            ),
            **final_response
        )
        yield chat_response

    @override
    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        config = ensure_config(config)
        return cast(
            "ChatGeneration",
            self.generate_prompt(
                [self._convert_input(input)],
                stop=stop,
                callbacks=config.get("callbacks"),
                tags=config.get("tags"),
                metadata=config.get("metadata"),
                run_name=config.get("run_name"),
                run_id=config.pop("run_id", None),
                **kwargs,
            ).generations[0][0],
        ).message

    @override
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        config = ensure_config(config)
        llm_result = await self.agenerate_prompt(
            [self._convert_input(input)],
            stop=stop,
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            **kwargs,
        )
        return cast("ChatGeneration", llm_result.generations[0][0]).message


Ollama = ChatOllama


class OllamaEmbeddings(BaseOllamaEmbeddings):
    pass
