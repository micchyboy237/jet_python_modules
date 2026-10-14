from dataclasses import dataclass
from typing import Any, TypeVar, overload

from browser_use.llm.messages import BaseMessage
from browser_use.llm.openai.chat import ChatOpenAI
from browser_use.llm.views import ChatInvokeCompletion
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class ChatLlamaCpp(ChatOpenAI):
    """
    Extends ChatOpenAI to support extra_body parameters for providers
    that require additional configuration (e.g., disabling thinking modes).
    """

    extra_body: dict[str, Any] | None = None

    @overload
    async def ainvoke(
        self, messages: list[BaseMessage], output_format: None = None, **kwargs: Any
    ) -> ChatInvokeCompletion[str]: ...

    @overload
    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T], **kwargs: Any
    ) -> ChatInvokeCompletion[T]: ...

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type[T] | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        """
        Invoke the model with the given messages, passing extra_body if provided.
        """
        # We need to replicate the logic from the parent class but inject extra_body
        # into the create call. Since the parent's ainvoke is complex, we can
        # either copy-paste and modify, or use a simpler approach by hooking into
        # the client creation.

        # However, the cleanest way without duplicating 100+ lines of code is
        # to override the specific part where the client is called.
        # But since ainvoke is a single large method in the parent, we will
        # implement a wrapper that modifies the behavior.

        # Actually, looking at the parent implementation, it builds model_params
        # and then calls self.get_client().chat.completions.create.
        # We can't easily intercept that without rewriting ainvoke.

        # Let's rewrite the necessary parts of ainvoke to support extra_body.

        from collections.abc import Iterable

        from browser_use.llm.base import is_reasoning_model
        from browser_use.llm.exceptions import (
            ModelOutputTruncatedError,
            ModelProviderError,
            ModelRateLimitError,
        )
        from browser_use.llm.openai.serializer import OpenAIMessageSerializer
        from browser_use.llm.schema import SchemaOptimizer
        from openai import APIConnectionError, APIStatusError, RateLimitError
        from openai.types.chat import ChatCompletionContentPartTextParam
        from openai.types.shared_params.response_format_json_schema import (
            JSONSchema,
            ResponseFormatJSONSchema,
        )

        openai_messages = OpenAIMessageSerializer.serialize_messages(messages)

        try:
            model_params: dict[str, Any] = {}
            if self.temperature is not None:
                model_params["temperature"] = self.temperature
            if self.frequency_penalty is not None:
                model_params["frequency_penalty"] = self.frequency_penalty
            if self.max_completion_tokens is not None:
                model_params["max_completion_tokens"] = self.max_completion_tokens
            if self.top_p is not None:
                model_params["top_p"] = self.top_p
            if self.seed is not None:
                model_params["seed"] = self.seed
            if self.service_tier is not None:
                model_params["service_tier"] = self.service_tier
            if is_reasoning_model(self.model, self.reasoning_models):
                model_params["reasoning_effort"] = self.reasoning_effort
                model_params.pop("temperature", None)
                model_params.pop("frequency_penalty", None)

            # Add extra_body if provided
            if self.extra_body is not None:
                model_params["extra_body"] = self.extra_body

            if output_format is None:
                response = await self.get_client().chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    **model_params,
                )
                choice = response.choices[0] if response.choices else None
                if choice is None:
                    base_url = str(self.base_url) if self.base_url is not None else None
                    hint = f" (base_url={base_url})" if base_url is not None else ""
                    raise ModelProviderError(
                        message=(
                            "Invalid OpenAI chat completion response: missing or empty `choices`."
                            " If you are using a proxy via `base_url`, ensure it implements the OpenAI"
                            " `/v1/chat/completions` schema and returns `choices` as a non-empty list."
                            f"{hint}"
                        ),
                        status_code=502,
                        model=self.name,
                    )
                usage = self._get_usage(response)
                return ChatInvokeCompletion(
                    completion=choice.message.content or "",
                    usage=usage,
                    stop_reason=choice.finish_reason,
                )
            else:
                response_format: JSONSchema = {
                    "name": "agent_output",
                    "strict": True,
                    "schema": SchemaOptimizer.create_optimized_json_schema(
                        output_format,
                        remove_min_items=self.remove_min_items_from_schema,
                        remove_defaults=self.remove_defaults_from_schema,
                    ),
                }
                if (
                    self.add_schema_to_system_prompt
                    and openai_messages
                    and openai_messages[0]["role"] == "system"
                ):
                    schema_text = f"\n<json_schema>\n{response_format}\n</json_schema>"
                    if isinstance(openai_messages[0]["content"], str):
                        openai_messages[0]["content"] += schema_text
                    elif isinstance(openai_messages[0]["content"], Iterable):
                        openai_messages[0]["content"] = list(
                            openai_messages[0]["content"]
                        ) + [
                            ChatCompletionContentPartTextParam(
                                text=schema_text, type="text"
                            )
                        ]

                if self.dont_force_structured_output:
                    response = await self.get_client().chat.completions.create(
                        model=self.model,
                        messages=openai_messages,
                        **model_params,
                    )
                else:
                    response = await self.get_client().chat.completions.create(
                        model=self.model,
                        messages=openai_messages,
                        response_format=ResponseFormatJSONSchema(
                            json_schema=response_format, type="json_schema"
                        ),
                        **model_params,
                    )

                choice = response.choices[0] if response.choices else None
                if choice is None:
                    base_url = str(self.base_url) if self.base_url is not None else None
                    hint = f" (base_url={base_url})" if base_url is not None else ""
                    raise ModelProviderError(
                        message=(
                            "Invalid OpenAI chat completion response: missing or empty `choices`."
                            " If you are using a proxy via `base_url`, ensure it implements the OpenAI"
                            " `/v1/chat/completions` schema and returns `choices` as a non-empty list."
                            f"{hint}"
                        ),
                        status_code=502,
                        model=self.name,
                    )
                if choice.finish_reason == "length":
                    cap = (
                        f"max_completion_tokens={self.max_completion_tokens}"
                        if self.max_completion_tokens is not None
                        else "the model's output token limit"
                    )
                    raise ModelOutputTruncatedError(
                        message=(
                            f"Model output was truncated at {cap};"
                            " the structured output is incomplete. Increase max_completion_tokens or request"
                            " shorter output."
                        ),
                        model=self.name,
                    )
                if choice.message.content is None:
                    raise ModelProviderError(
                        message="Failed to parse structured output from model response",
                        status_code=500,
                        model=self.name,
                    )
                usage = self._get_usage(response)
                parsed = output_format.model_validate_json(choice.message.content)
                return ChatInvokeCompletion(
                    completion=parsed,
                    usage=usage,
                    stop_reason=choice.finish_reason,
                )
        except ModelProviderError:
            raise
        except RateLimitError as e:
            raise ModelRateLimitError(message=e.message, model=self.name) from e
        except APIConnectionError as e:
            raise ModelProviderError(message=str(e), model=self.name) from e
        except APIStatusError as e:
            raise ModelProviderError(
                message=e.message, status_code=e.status_code, model=self.name
            ) from e
        except Exception as e:
            raise ModelProviderError(message=str(e), model=self.name) from e
