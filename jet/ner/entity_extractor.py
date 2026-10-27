# jet/ner/entity_extractor.py
import json
import os
from typing import List, Type, TypeVar

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_utils_observed import chat
from jet.logger import logger
from jet_telemetry import chain, get_service_name, redact
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace as otel_trace
from pydantic import BaseModel, Field, ValidationError

T = TypeVar("T", bound=BaseModel)

ENTITIES_EXTRACTION_TEXT_TEMPLATE = """
Priority Information:
{priority}

Additional Context:
{others}
"""


class KeyFilterResult(BaseModel):
    """
    Structured output to determine which keys from 'other_data'
    should be kept because they provide unique information not present in 'priority_data'.
    """

    keys_to_keep: List[str] = Field(
        description="List of keys from other_data that contain unique information NOT found in priority_data."
    )
    reasoning: str = Field(
        description="Brief explanation of why certain keys were removed or kept."
    )


def filter_conflicting_keys(
    priority_text: str,
    other_data: dict,
    model_name: str = "qwen3.5-uncensored:2b",
    temperature: float = 0.0,
) -> dict:
    """
    Uses an LLM to filter out keys from other_data if their information
    is already present or contradicted by priority_text.

    Args:
        priority_text: The text derived from priority fields (title, details).
        other_data: A dictionary of key-value pairs from non-priority fields.
        model_name: Model to use for filtering.
        temperature: Sampling temperature (0.0 recommended for deterministic filtering).

    Returns:
        A filtered dictionary containing only unique/non-conflicting keys.
    """
    if not other_data:
        return {}

    # Prepare the other_data for the prompt
    other_items_str = "\n".join([f"- {k}: {v}" for k, v in other_data.items()])

    prompt = f"""
You are an intelligent data deduplication assistant.
Your goal is to identify which pieces of information in 'Other Data' are REDUNDANT or CONFLICTING with 'Priority Data'.

Priority Data (High Confidence):
{priority_text}

Other Data (Lower Confidence / Metadata):
{other_items_str}

Instructions:
1. Analyze the 'Priority Data'. This is the ground truth.
2. Look at each item in 'Other Data'.
3. If the information in an 'Other Data' item is already clearly stated in 'Priority Data', mark it for removal.
4. If the 'Other Data' item provides NEW, unique information not found in 'Priority Data', mark it to keep.
5. If there is a direct conflict, trust 'Priority Data' and remove the conflicting 'Other Data' item.

Return a JSON object with:
- keys_to_keep: A list of keys from 'Other Data' that should be retained.
- reasoning: A brief explanation.
"""

    try:
        result = chat(
            prompt_or_messages=[{"role": "user", "content": prompt}],
            model=model_name,
            response_format=KeyFilterResult,
            temperature=temperature,
            max_tokens=500,
            project_name="entity-key-filter",
        )

        if result.structured and result.structured.success:
            filtered_result: KeyFilterResult = result.structured.parsed
            kept_keys = filtered_result.keys_to_keep

            # Filter the original dictionary
            filtered_dict = {k: v for k, v in other_data.items() if k in kept_keys}
            logger.debug(f"Key Filter Reasoning: {filtered_result.reasoning}")
            logger.debug(f"Kept keys: {kept_keys}")
            return filtered_dict
        else:
            logger.warning(
                "Key filtering failed structured validation. Returning original other_data."
            )
            return other_data

    except Exception as e:
        logger.error(f"Error during key filtering LLM call: {e}")
        # Fallback: Return original data if LLM fails to avoid data loss
        return other_data


def build_entity_extraction_text(job: dict) -> str:
    """
    Dynamically build a rich context string for entity extraction by including
    ALL available non-empty job fields except 'id'. Handles strings, lists, dicts,
    and numeric types with appropriate formatting.

    NEW: Filters 'other_keys' using an LLM to remove redundancy with 'priority_keys'.
    """
    EXCLUDED_KEYS = {"id", "entities"}
    priority_keys = ["title", "details"]

    # Separate priority and other data
    priority_data = {}
    other_data = {}

    for key in job.keys():
        if key in EXCLUDED_KEYS:
            continue

        value = job.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue

        if key in priority_keys:
            priority_data[key] = value
        else:
            other_data[key] = value

    # Build Priority Text
    priority_parts = []
    for key in priority_keys:
        value = priority_data.get(key)
        if value:
            label = "Overview" if key == "details" else key.replace("_", " ").title()
            formatted = str(value).strip()
            if key == "details":
                priority_parts.append(f"{label}:\n{formatted}")
            else:
                priority_parts.append(f"{label}: {formatted}")

    priority_parts_str = "\n".join(priority_parts)

    # --- NEW: Filter Other Keys ---
    # Only perform filtering if we have both priority and other data
    if priority_parts_str and other_data:
        logger.debug("Running conflict filter for other keys...")
        # Use a low temperature for deterministic filtering
        filtered_other_data = filter_conflicting_keys(
            priority_text=priority_parts_str, other_data=other_data, temperature=0.0
        )
    else:
        filtered_other_data = other_data

    # Build Other Text from Filtered Data
    other_parts = []
    for key, value in filtered_other_data.items():
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
                other_parts.append(f"{label}: {', '.join(cleaned)}")
        elif isinstance(value, dict):
            dict_parts = []
            for dk, dv in value.items():
                if dv is not None and str(dv).strip():
                    dk_label = str(dk).replace("_", " ").title()
                    dict_parts.append(f"{dk_label}: {str(dv).strip()}")
            if dict_parts:
                other_parts.append(f"{label}: {' | '.join(dict_parts)}")
        else:
            formatted = str(value).strip()
            if formatted:
                other_parts.append(f"{label}: {formatted}")

    other_parts_str = "\n".join(other_parts)

    return ENTITIES_EXTRACTION_TEXT_TEMPLATE.format(
        priority=priority_parts_str, others=other_parts_str
    )


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
                enable_thinking=attempts_made > 2,
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


if __name__ == "__main__":
    from jet.ner.main._main_entity_extractor import main

    main()
