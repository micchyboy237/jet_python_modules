# JetScripts/libs/stanza/common/run_data_objects.py
"""
Run examples demonstrating stanza data object property extensions.
Results are saved to JSON files in the same directory.
"""
import logging
import os
from typing import Any, Dict


from jet.file.utils import save_file
import shutil

from jet.libs.stanza.common.extract_data_objects import extract_backpointer, extract_getter, extract_readonly, extract_setter_getter

OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# --------------------------------------------------------------------------- #
# Logging & Progress
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def _save_result(name: str, data: Any) -> None:
    """Persist a single example result as pretty-printed JSON."""
    path = os.path.join(os.path.dirname(__file__), f"{OUTPUT_DIR}/{name}.json")
    save_file(data, path)

# --------------------------------------------------------------------------- #
# Example functions (return typed values)
# --------------------------------------------------------------------------- #

def example_readonly() -> Dict[str, Any]:
    """Demonstrate a read-only document property."""
    return extract_readonly(
        input_text="This is a test document. Pretty cool!",
        property_name="some_property",
        property_value=123,
    )

def example_getter() -> Dict[str, Any]:
    """Show a derived word property combining UPOS+XPOS."""
    return extract_getter(
        input_text="This is a test document. Pretty cool!",
    )

def example_setter_getter() -> Dict[str, Any]:
    """Illustrate a sentence property with custom setter/getter."""
    return extract_setter_getter(
        input_text="This is a test document. Pretty cool!",
        prop_name="classname",
        set_good_value="good",
        set_bad_internal=2,
    )

def example_backpointer() -> Dict[str, Any]:
    """Verify back-pointers from words/tokens/entities to their sentence."""
    return extract_backpointer(
        input_text="Chris Manning wrote a sentence. Then another.",
    )

# --------------------------------------------------------------------------- #
# Main runner
# --------------------------------------------------------------------------- #
def main() -> None:
    examples = [
        example_readonly,
        example_getter,
        example_setter_getter,
        example_backpointer,
    ]
    total = len(examples)
    for idx, func in enumerate(examples, start=1):
        logger.info("Running %s (%d/%d)...", func.__name__, idx, total)
        result = func()
        _save_result(func.__name__, result)
    logger.info("All %d examples completed.", total)

if __name__ == "__main__":
    main()