import os
from datetime import datetime, timedelta
from typing import Type, TypeVar

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_utils_observed import chat
from jet.logger import logger
from jet_telemetry import chain, get_service_name, redact
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace as otel_trace
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def filter_jobs(jobs: list[dict], days: int, include_existing: bool) -> list[dict]:
    filtered = []
    cutoff_date = None
    if days > 0:
        cutoff_date = datetime.now().astimezone() - timedelta(days=days)
        logger.info(
            f"Filtering jobs posted after {cutoff_date.strftime('%Y-%m-%d %H:%M %Z')}"
        )
    skipped_old = skipped_existing = skipped_invalid = 0
    for job in jobs:
        if not include_existing and job.get("entities") is not None:
            skipped_existing += 1
            continue
        if cutoff_date is not None:
            posted_str = job.get("posted_date")
            if not posted_str:
                skipped_invalid += 1
                continue
            try:
                posted_dt = datetime.fromisoformat(
                    str(posted_str).replace("Z", "+00:00")
                )
                if posted_dt.tzinfo is None:
                    posted_dt = posted_dt.astimezone()
                if posted_dt < cutoff_date:
                    skipped_old += 1
                    continue
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Skipping job '{job.get('id', '?')}' – bad date '{posted_str}': {e}"
                )
                skipped_invalid += 1
                continue
        filtered.append(job)
    logger.debug(
        f"Filter: {len(filtered)} eligible | {skipped_old} old | "
        f"{skipped_existing} existing | {skipped_invalid} invalid dates"
    )
    return filtered


def build_entity_extraction_text(job: dict) -> str:
    """
    Dynamically build a rich context string for entity extraction by including
    ALL available non-empty job fields except 'id'. Handles strings, lists, dicts,
    and numeric types with appropriate formatting.
    """
    EXCLUDED_KEYS = {"id", "entities"}
    parts = []

    # Prioritize title and details at the top for consistent LLM attention
    priority_keys = ["title", "details"]
    for key in priority_keys:
        value = job.get(key)
        if value is not None and str(value).strip():
            label = "Overview" if key == "details" else key.replace("_", " ").title()
            formatted = str(value).strip()
            if key == "details":
                parts.append(f"{label}:\n{formatted}")
            else:
                parts.append(f"{label}: {formatted}")

    # Process all remaining keys dynamically
    for key, value in job.items():
        if key in EXCLUDED_KEYS or key in priority_keys:
            continue
        if value is None:
            continue

        label = key.replace("_", " ").title()

        if isinstance(value, list):
            cleaned = [
                str(item).strip()
                for item in value
                if item is not None and str(item).strip()
            ]
            if cleaned:
                parts.append(f"{label}: {', '.join(cleaned)}")

        elif isinstance(value, dict):
            # Flatten dict into readable key-value pairs
            dict_parts = []
            for dk, dv in value.items():
                if dv is not None and str(dv).strip():
                    dk_label = str(dk).replace("_", " ").title()
                    dict_parts.append(f"{dk_label}: {str(dv).strip()}")
            if dict_parts:
                parts.append(f"{label}: {' | '.join(dict_parts)}")

        else:
            # Handle str, int, float, bool, date, etc.
            formatted = str(value).strip()
            if formatted:
                parts.append(f"{label}: {formatted}")

    return "\n".join(parts)


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
