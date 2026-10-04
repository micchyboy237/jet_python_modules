# jet/ner/entity_extractor.py
import json
import os
from typing import Type, TypeVar

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_utils_observed import chat
from jet.logger import logger
from jet_telemetry import chain, get_service_name, redact
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace as otel_trace
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def _build_correction_prompt(
    original_text: str, error_details: str, model_class: Type[T]
) -> str:
    """
    Builds a targeted correction prompt that includes valid enum values
    to prevent the LLM from repeating the same invalid output.
    """
    # Extract valid enum values from the Pydantic model schema
    schema = model_class.model_json_schema()
    enum_hints = []

    for field_name, props in schema.get("properties", {}).items():
        if "enum" in props:
            valid_vals = ", ".join(f"'{v}'" for v in props["enum"])
            enum_hints.append(f"- {field_name}: MUST be one of [{valid_vals}]")

    enum_section = ""
    if enum_hints:
        enum_section = "\n\nVALID ENUM VALUES (STRICTLY USE THESE):\n" + "\n".join(
            enum_hints
        )

    return (
        f"{original_text}\n\n"
        f"=== VALIDATION FAILED ===\n"
        f"Your previous response was INVALID.\n"
        f"ERRORS:\n{error_details}\n"
        f"{enum_section}\n\n"
        f"INSTRUCTION: Return ONLY corrected valid JSON. Do NOT repeat the invalid values."
    )


@chain(name="extract-entities-chain")
def extract_entities_from_text(
    text: str,
    model_class: Type[T],
    temperature: float = 0.3,
    timeout: float = 30.0,
    max_retries: int = 5,  # Allow 1 initial attempt + 5 retries
) -> T:
    """
    Extract structured entities from text using local LLM with full observability.
    Includes a self-correction loop for Pydantic validation errors.
    """
    model_name = os.getenv("LLAMA_CPP_LLM_MODEL", LLM_MODEL)
    span = otel_trace.get_current_span()

    # Initial System Prompt
    # system_prompt = (
    #     "You are an expert entity extraction assistant. Extract information strictly "
    #     "according to the provided JSON schema. Return ONLY valid JSON."
    # )

    last_error = None
    attempts_made = 0

    for attempt in range(max_retries):
        attempts_made += 1
        try:
            current_service = get_service_name()

            # If this is a retry, append error context to the user message
            if last_error:
                user_content = _build_correction_prompt(text, last_error, model_class)
                logger.debug(
                    f"Retry {attempt} with correction prompt for {model_class.__name__}"
                )
            else:
                user_content = text

            # Reduce temperature on retries to encourage strict adherence
            current_temp = max(0.0, temperature - (0.1 * attempt))

            result = chat(
                prompt_or_messages=[
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                model=model_name,
                response_format=model_class,
                temperature=current_temp,
                project_name=current_service,
                max_tokens=2000,
                enable_thinking=attempts_made > 1,
            )

            # Check if structured output was successful
            if not result.structured or not result.structured.success:
                error_msg = (
                    result.structured.error
                    if result.structured
                    else "Unknown parsing error"
                )
                validation_errors = (
                    result.structured.validation_errors
                    if result.structured and result.structured.validation_errors
                    else []
                )

                # Format errors clearly for the next iteration
                formatted_errors = (
                    json.dumps(validation_errors, indent=2)
                    if validation_errors
                    else error_msg
                )
                raise ValueError(formatted_errors)

            parsed_entity = result.structured.parsed

            # Log success if it was a retry
            if attempt > 0:
                logger.info(
                    f"Entity extraction succeeded on attempt {attempts_made} after correction."
                )

            if span.is_recording() and parsed_entity:
                span.set_attribute("extraction.attempts_needed", attempts_made)
                span.set_attribute("extraction.status", "success")
                output_summary = str(parsed_entity.model_dump(mode="json"))[:500]
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, redact(output_summary))

            return parsed_entity

        except ValidationError as e:
            # Capture Pydantic validation errors specifically
            last_error = str(e.errors())
            logger.warning(
                f"Attempt {attempts_made} failed Pydantic validation: {last_error[:200]}..."
            )
            continue

        except ValueError as e:
            # Capture structured output parsing failures
            last_error = str(e)
            logger.warning(
                f"Attempt {attempts_made} failed structured parsing: {last_error[:200]}..."
            )
            continue

        except Exception as e:
            # Handle other errors (network issues, etc.)
            last_error = str(e)
            logger.warning(
                f"Attempt {attempts_made} failed with unexpected error: {last_error}"
            )
            continue

    # If all attempts fail
    if span.is_recording():
        span.set_attribute("extraction.attempts_needed", attempts_made)
        span.set_attribute("extraction.status", "failed")
        span.record_exception(Exception(last_error))
        span.set_status(
            otel_trace.status.StatusCode.ERROR,
            "Max retries reached for entity extraction",
        )

    logger.error(
        f"Failed to extract entities after {max_retries} attempts. Last error: {last_error}"
    )
    raise Exception(
        f"Entity extraction failed after {max_retries} retries. Last error: {last_error}"
    )
