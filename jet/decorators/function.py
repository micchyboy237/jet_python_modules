from functools import wraps
from typing import Callable, TypeVar, Union, Sequence

from jet.logger import logger
from jet.logger.timer import sleep_countdown

F = TypeVar("F", bound=Callable[..., list[float]])


# Define the retry decorator
def retry_on_error(max_retries: int = 5, delay: int = 5):
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {
                                attempt + 1} failed: {e}. Retrying in {delay} seconds..."
                        )
                        sleep_countdown(delay)
                    else:
                        logger.error(
                            "Max retries reached. Raising the exception.")
                        raise
        return wrapper  # type: ignore
    return decorator


def llm_chat_callback() -> Callable:
    def wrap(f: Callable) -> Callable:
        @contextmanager
        def wrapper_logic(_self: Any) -> Generator[CallbackManager, None, None]:
            callback_manager = getattr(_self, "callback_manager", None)
            if not isinstance(callback_manager, CallbackManager):
                _self.callback_manager = CallbackManager()

            yield _self.callback_manager  # type: ignore

        async def wrapped_async_llm_chat(
            _self: Any, messages: Sequence[ChatMessage], **kwargs: Any
        ) -> Any:
            with wrapper_logic(_self) as callback_manager, callback_manager.as_trace(
                "chat"
            ):
                span_id = active_span_id.get()
                model_dict = _self.to_dict()
                model_dict.pop("api_key", None)
                dispatcher.event(
                    LLMChatStartEvent(
                        model_dict=model_dict,
                        messages=messages,
                        additional_kwargs=kwargs,
                        span_id=span_id,
                    )
                )
                event_id = callback_manager.on_event_start(
                    CBEventType.LLM,
                    payload={
                        EventPayload.MESSAGES: messages,
                        EventPayload.ADDITIONAL_KWARGS: kwargs,
                        EventPayload.SERIALIZED: _self.to_dict(),
                    },
                )
                try:
                    f_return_val = await f(_self, messages, **kwargs)
                except BaseException as e:
                    callback_manager.on_event_end(
                        CBEventType.LLM,
                        payload={EventPayload.EXCEPTION: e},
                        event_id=event_id,
                    )
                    raise
                if isinstance(f_return_val, AsyncGenerator):
                    # intercept the generator and add a callback to the end
                    async def wrapped_gen() -> ChatResponseAsyncGen:
                        last_response = None
                        try:
                            async for x in f_return_val:
                                dispatcher.event(
                                    LLMChatInProgressEvent(
                                        messages=messages,
                                        response=x,
                                        span_id=span_id,
                                    )
                                )
                                yield cast(ChatResponse, x)
                                last_response = x
                        except BaseException as exception:
                            callback_manager.on_event_end(
                                CBEventType.LLM,
                                payload={EventPayload.EXCEPTION: exception},
                                event_id=event_id,
                            )
                            dispatcher.event(
                                ExceptionEvent(
                                    exception=exception,
                                    span_id=span_id,
                                )
                            )
                            raise
                        callback_manager.on_event_end(
                            CBEventType.LLM,
                            payload={
                                EventPayload.MESSAGES: messages,
                                EventPayload.RESPONSE: last_response,
                            },
                            event_id=event_id,
                        )
                        dispatcher.event(
                            LLMChatEndEvent(
                                messages=messages,
                                response=last_response,
                                span_id=span_id,
                            )
                        )

                    return wrapped_gen()
                else:
                    callback_manager.on_event_end(
                        CBEventType.LLM,
                        payload={
                            EventPayload.MESSAGES: messages,
                            EventPayload.RESPONSE: f_return_val,
                        },
                        event_id=event_id,
                    )
                    dispatcher.event(
                        LLMChatEndEvent(
                            messages=messages,
                            response=f_return_val,
                            span_id=span_id,
                        )
                    )

            return f_return_val

        def wrapped_llm_chat(
            _self: Any, messages: Sequence[ChatMessage], **kwargs: Any
        ) -> Any:
            with wrapper_logic(_self) as callback_manager, callback_manager.as_trace(
                "chat"
            ):
                span_id = active_span_id.get()
                model_dict = _self.to_dict()
                model_dict.pop("api_key", None)
                dispatcher.event(
                    LLMChatStartEvent(
                        model_dict=model_dict,
                        messages=messages,
                        additional_kwargs=kwargs,
                        span_id=span_id,
                    )
                )
                event_id = callback_manager.on_event_start(
                    CBEventType.LLM,
                    payload={
                        EventPayload.MESSAGES: messages,
                        EventPayload.ADDITIONAL_KWARGS: kwargs,
                        EventPayload.SERIALIZED: _self.to_dict(),
                    },
                )
                try:
                    f_return_val = f(_self, messages, **kwargs)
                except BaseException as e:
                    callback_manager.on_event_end(
                        CBEventType.LLM,
                        payload={EventPayload.EXCEPTION: e},
                        event_id=event_id,
                    )
                    raise
                if isinstance(f_return_val, Generator):
                    # intercept the generator and add a callback to the end
                    def wrapped_gen() -> ChatResponseGen:
                        last_response = None
                        try:
                            for x in f_return_val:
                                dispatcher.event(
                                    LLMChatInProgressEvent(
                                        messages=messages,
                                        response=x,
                                        span_id=span_id,
                                    )
                                )
                                yield cast(ChatResponse, x)
                                last_response = x
                        except BaseException as exception:
                            callback_manager.on_event_end(
                                CBEventType.LLM,
                                payload={EventPayload.EXCEPTION: exception},
                                event_id=event_id,
                            )
                            dispatcher.event(
                                ExceptionEvent(
                                    exception=exception,
                                    span_id=span_id,
                                )
                            )
                            raise
                        callback_manager.on_event_end(
                            CBEventType.LLM,
                            payload={
                                EventPayload.MESSAGES: messages,
                                EventPayload.RESPONSE: last_response,
                            },
                            event_id=event_id,
                        )
                        dispatcher.event(
                            LLMChatEndEvent(
                                messages=messages,
                                response=last_response,
                                span_id=span_id,
                            )
                        )

                    return wrapped_gen()
                else:
                    callback_manager.on_event_end(
                        CBEventType.LLM,
                        payload={
                            EventPayload.MESSAGES: messages,
                            EventPayload.RESPONSE: f_return_val,
                        },
                        event_id=event_id,
                    )
                    dispatcher.event(
                        LLMChatEndEvent(
                            messages=messages,
                            response=f_return_val,
                            span_id=span_id,
                        )
                    )

            return f_return_val

        async def async_dummy_wrapper(_self: Any, *args: Any, **kwargs: Any) -> Any:
            return await f(_self, *args, **kwargs)

        def dummy_wrapper(_self: Any, *args: Any, **kwargs: Any) -> Any:
            return f(_self, *args, **kwargs)

        # check if already wrapped
        is_wrapped = getattr(f, "__wrapped__", False)
        if not is_wrapped:
            f.__wrapped__ = True  # type: ignore

        # Update the wrapper function to look like the wrapped function.
        # See e.g. https://github.com/python/cpython/blob/0abf997e75bd3a8b76d920d33cc64d5e6c2d380f/Lib/functools.py#L57
        for attr in (
            "__module__",
            "__name__",
            "__qualname__",
            "__doc__",
            "__annotations__",
            "__type_params__",
        ):
            if v := getattr(f, attr, None):
                setattr(async_dummy_wrapper, attr, v)
                setattr(wrapped_async_llm_chat, attr, v)
                setattr(dummy_wrapper, attr, v)
                setattr(wrapped_llm_chat, attr, v)

        if asyncio.iscoroutinefunction(f):
            if is_wrapped:
                return async_dummy_wrapper
            else:
                return wrapped_async_llm_chat
        else:
            if is_wrapped:
                return dummy_wrapper
            else:
                return wrapped_llm_chat

    return wrap


__all__ = [
    "retry_on_error",
    "llm_chat_callback",
]
