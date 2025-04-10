import json
import random
from typing import Optional
from jet.validation import schema_validate_json, ValidationResponse
from jet.utils.markdown import extract_json_block_content
from jet.logger import logger

MODEL = "llama3.1"
PROMPT_TEMPLATE = """
Schema:
[schema]

JSON to correct:
[prompt]

Validation errors:
[errors]

Fix JSON based on schema and errors:
"""

SYSTEM = f"""
System:
You are a JSON corrector. Analyze the provided schema, JSON to validate and validation errors then provide a corrected JSON object that is valid according to the schema.
Surround the generated object with JSON block ```json\n<generated_json>\n```.
Generated response should only have one JSON block.
Do not include any other text in the response except the JSON block.

Data guidelines:
- Follow validation errors to correct the JSON object
- Ensure the corrected JSON object has valid syntax
- Enumerated values should match case and spelling in the schema
""".strip()


def validate_json(json_string: str | dict, schema: Optional[dict] = None, model: str = MODEL, attempt: int = 1, max_attempts: int = 10, original_json: Optional[str] = None, generated_error: Optional[Exception] = None) -> dict:
    from jet.actions.generation import call_ollama_chat

    if isinstance(json_string, dict):
        json_string = json.dumps(json_string)

    if original_json is None:
        original_json = json_string  # Save the original JSON for comparison

    validation_response: ValidationResponse = schema_validate_json(
        json_string, schema)

    if attempt > max_attempts:
        logger.error("Max recursive validation attempts reached.")
        return {"data": json_string, "corrected": False, "is_valid": False, "errors": validation_response["errors"]}

    logger.info(f"Validation attempt {attempt}")

    if validation_response["is_valid"] and not validation_response["errors"]:
        logger.success(f"Valid JSON on attempt {attempt}")
        return {"data": validation_response["data"], "corrected": json_string != original_json, "is_valid": True}

    logger.error(f"Invalid JSON on attempt {attempt}")
    error_prompt = generated_error if generated_error else validation_response["errors"]
    if isinstance(error_prompt, list):
        error_prompt = str(error_prompt)

    prompt = f"{PROMPT_TEMPLATE.replace('[schema]', str(schema)).replace(
        '[prompt]', json_string).replace('[errors]', str(error_prompt))}"

    try:
        output = ""
        response = call_ollama_chat(
            prompt,
            model=model,
            system=SYSTEM,
            stream=True,
            options={
                "seed": random.randint(1, 9999),
                # "num_keep": 0,
                "num_predict": -1,
                "temperature": 0,
            },
        )
        for chunk in response:
            output += chunk
        extracted_result = extract_json_block_content(output)
        logger.info(f"Extracted JSON content:\n{extracted_result}")
        return validate_json(extracted_result, schema, model, attempt + 1, max_attempts, original_json)
    except Exception as generated_error:
        logger.error(f"Failed to decode AI response: {generated_error}")
        return validate_json(json_string, schema, model, attempt + 1, max_attempts, original_json, generated_error)
