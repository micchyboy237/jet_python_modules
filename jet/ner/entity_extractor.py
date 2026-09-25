import os
from typing import Type, TypeVar

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_utils_observed import chat
from jet.logger import logger
from jet_telemetry import chain, get_service_name, redact
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace as otel_trace
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@chain(name="extract-entities-chain")
def extract_entities_from_text(
    text: str,
    model_class: Type[T],
    temperature: float = 0.3,
    timeout: float = 30.0,
) -> T:
    """
    Extract structured entities from text using local LLM with full observability.

    Uses jet.adapters.llama_cpp.llm_utils_observed.chat to handle:
    - Automatic tracing (LLM span + Structured Output evaluation)
    - Pydantic validation
    - Token counting

    Args:
        text: The input text from which to extract entities.
        model_class: A Pydantic BaseModel class defining the structure.
        temperature: Sampling temperature.
        timeout: Not directly used by chat() but kept for signature compatibility.

    Returns:
        An instance of the provided Pydantic model with extracted entities.

    Raises:
        Exception: If extraction or validation fails.
    """
    model_name = os.getenv("LLAMA_CPP_LLM_MODEL", LLM_MODEL)

    # Capture input in the parent Chain span
    span = otel_trace.get_current_span()
    if span.is_recording():
        span.set_attribute(
            SpanAttributes.INPUT_VALUE,
            redact(text[:500] + "..." if len(text) > 500 else text),
        )
        span.set_attribute("input.text_length", len(text))

    system_prompt = (
        "You are an entity extraction assistant. Extract information strictly "
        "according to the provided JSON schema. Return ONLY valid JSON."
    )

    try:
        current_service = get_service_name()

        # Use the observed chat utility which handles streaming, parsing, and tracing
        result = chat(
            prompt_or_messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            model=model_name,
            response_format=model_class,  # Automatically injects schema and validates
            temperature=temperature,
            project_name=current_service,  # Groups traces in Phoenix
            max_tokens=2000,
            extra_body_params={"chat_template_kwargs": {"enable_thinking": False}},
        )

        # Check if structured parsing succeeded
        if not result.structured or not result.structured.success:
            error_msg = (
                result.structured.error
                if result.structured
                else "Unknown parsing error"
            )
            validation_errors = (
                result.structured.validation_errors if result.structured else []
            )
            logger.error(f"Structured output failed: {error_msg} | {validation_errors}")
            raise ValueError(f"Entity extraction validation failed: {error_msg}")

        parsed_entity = result.structured.parsed

        # Capture output in the parent Chain span
        if span.is_recording() and parsed_entity:
            # Dump only a summary or the first few fields to avoid huge spans
            output_summary = str(parsed_entity.model_dump(mode="json"))[:500]
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, redact(output_summary))

        return parsed_entity

    except Exception as e:
        if span.is_recording():
            span.record_exception(e)
            span.set_status(otel_trace.status.StatusCode.ERROR, str(e))
        logger.error(f"Failed to extract entities: {e}")
        raise


if __name__ == "__main__":
    from jet.ner.main._main_entity_extractor import main

    main()
