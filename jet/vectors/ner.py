from typing import Optional
from jet.executor.command import run_command
from jet.logger import logger
import json
import os

NER_MODEL = "urchade/gliner_small-v2.1"
NER_STYLE = "ent"
NER_LABELS = ["role", "application", "technology stack", "qualifications"]


def determine_chunk_size(text: str) -> int:
    """Dynamically set chunk size based on text length."""
    length = len(text)
    if length < 1000:
        return 250
    elif length < 3000:
        return 350
    else:
        return 500


def extract_named_entities(texts: list[str], *, model: str = NER_MODEL,
                           labels: list[str] = NER_LABELS,
                           style: str = NER_STYLE, chunk_size: Optional[int] = None) -> list[dict[str, str]]:
    """Extract named entities from the given list of texts."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    execute_file = os.path.join(current_dir, "ner_execute_file.py")

    # Prepare texts in a way that they can be processed as a list
    logger.info(f"Dynamic chunk size set to: {chunk_size}")
    labels_json = json.dumps(labels)
    texts_json = json.dumps(texts)

    command_separator = "<sep>"
    command_args = [
        "python",
        execute_file,
        model,
        texts_json,
        labels_json,
        style,
    ]
    command = command_separator.join(command_args)

    error_lines = []
    debug_lines = []
    entities = []

    logger.newline()
    logger.debug("Extracted Entities:")

    for line in run_command(command, separator=command_separator):
        if line.startswith('error: '):
            message = line[7:-2]
            error_lines.append(message)
            logger.error(message)
        elif line.startswith('result: '):
            message = line[8:-2]
            try:
                result = json.loads(message)

                entities.append(result)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON result: {message}")
        else:
            message = line[6:-2]
            debug_lines.append(message)
            logger.debug(message)

    if not entities and debug_lines:
        logger.debug("\n".join(debug_lines))
        logger.error("\n".join(error_lines))

    return entities
