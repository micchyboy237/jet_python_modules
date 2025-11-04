# JetScripts/libs/stanza/common/run_data_objects.py
"""
Run examples demonstrating stanza data object property extensions.
Results are saved to JSON files in the same directory.
"""
import logging
from typing import Any, Dict, Optional

import stanza
from stanza.models.common.doc import Document, Sentence, Word
from stanza.tests import TEST_MODELS_DIR

# --------------------------------------------------------------------------- #
# Logging & Progress
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Pipeline fixture (shared across examples)
# --------------------------------------------------------------------------- #
_pipeline: Optional[stanza.Pipeline] = None

def _get_pipeline() -> stanza.Pipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Building Stanza pipeline (en)...")
        _pipeline = stanza.Pipeline(dir=TEST_MODELS_DIR, lang="en", verbose=False)
        logger.info("Pipeline ready.")
    return _pipeline

# --------------------------------------------------------------------------- #
# Extraction functions (return typed values, reusable via args)
# --------------------------------------------------------------------------- #

def extract_readonly(
    input_text: str = "This is a test document. Pretty cool!",
    property_name: str = "some_property",
    property_value: Any = 123,
    pipeline: Optional[stanza.Pipeline] = None
) -> Dict[str, Any]:
    """Demonstrate a read-only document property."""
    if not hasattr(Document, property_name):
        Document.add_property(property_name, property_value)
    nlp = pipeline if pipeline is not None else _get_pipeline()
    doc = nlp(input_text)
    return {
        "input_text": input_text,
        property_name: getattr(doc, property_name),
        "attempt_set": "(raises ValueError)",
    }

def extract_getter(
    input_text: str = "This is a test document. Pretty cool!",
    prop_name: str = "upos_xpos_word",
    getter_fn = lambda self: f"{self.upos}_{self.xpos}_{self.text}",
    pipeline: Optional[stanza.Pipeline] = None
) -> Dict[str, Any]:
    """Show a derived word property combining UPOS+XPOS+word text."""
    if not hasattr(Word, prop_name):
        Word.add_property(prop_name, getter=getter_fn)
    nlp = pipeline if pipeline is not None else _get_pipeline()
    doc = nlp(input_text)
    tuples = tuple(
        tuple(getattr(word, prop_name) for word in sent.words) for sent in doc.sentences
    )
    # Return key is built from the *actual* prop_name used
    return_key = f"{prop_name}_per_sentence"
    return {
        "input_text": input_text,
        return_key: tuples,
    }

def extract_setter_getter(
    input_text: str = "This is a test document. Pretty cool!",
    prop_name: str = "classname",
    int2str: Optional[Dict[int, str]] = None,
    str2int: Optional[Dict[str, int]] = None,
    pipeline: Optional[stanza.Pipeline] = None,
    set_good_value: str = "good",
    set_bad_internal: int = 2,
    ) -> Dict[str, Any]:
    """Illustrate a sentence property with custom setter/getter."""
    int2str = int2str if int2str is not None else {0: "ok", 1: "good", 2: "bad"}
    str2int = str2int if str2int is not None else {"ok": 0, "good": 1, "bad": 2}

    def setter(self, value: str) -> None:
        self._classname = str2int[value]

    if not hasattr(Sentence, prop_name):
        Sentence.add_property(
            prop_name,
            getter=lambda self: (
                int2str[self._classname] if getattr(self, "_classname", None) is not None else None
            ),
            setter=setter,
        )
    nlp = pipeline if pipeline is not None else _get_pipeline()
    doc = nlp(input_text)
    sent = doc.sentences[0]
    setattr(sent, prop_name, set_good_value)
    internal = getattr(sent, "_classname")
    setattr(sent, "_classname", set_bad_internal)
    final = getattr(sent, prop_name)
    return {
        "input_text": input_text,
        "after_set_good": {"_classname": internal, prop_name: set_good_value},
        "after_set_bad": {"_classname": set_bad_internal, prop_name: final},
    }

def extract_backpointer(
    input_text: str = "Chris Manning wrote a sentence. Then another.",
    pipeline: Optional[stanza.Pipeline] = None
) -> Dict[str, Any]:
    """Verify back-pointers from words/tokens/entities to their sentence."""
    nlp = pipeline if pipeline is not None else _get_pipeline()
    doc = nlp(input_text)
    ent = doc.ents[0] if doc.ents else None
    first_word = next(doc.iter_words())
    last_token = list(doc.iter_tokens())[-1]
    return {
        "input_text": input_text,
        "entity_sentence_index": doc.sentences.index(ent.sent) if ent else None,
        "first_word_sentence_index": doc.sentences.index(first_word.sent),
        "last_token_sentence_index": doc.sentences.index(last_token.sent),
    }
