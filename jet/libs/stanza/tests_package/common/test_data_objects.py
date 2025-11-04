# tests/test_extract_data_objects.py
from typing import Any, Dict, Tuple

import pytest
from stanza.models.common.doc import Document, Sentence, Word

from jet.libs.stanza.common.extract_data_objects import (
    extract_backpointer,
    extract_getter,
    extract_readonly,
    extract_setter_getter,
)


@pytest.fixture(scope="module")
def nlp_pipeline():
    """Shared Stanza pipeline for all tests to avoid repeated loading."""
    from jet.libs.stanza.common.extract_data_objects import _get_pipeline

    pipeline = _get_pipeline()
    yield pipeline
    # Reset global pipeline to avoid side effects across test runs
    from jet.libs.stanza.common.extract_data_objects import _pipeline

    _pipeline = None


@pytest.fixture(autouse=True)
def reset_properties():
    """Ensure Word/Sentence/Document properties are reset before each test."""
    yield
    # Clean up any added properties
    for cls in (Word, Sentence, Document):
        for attr in list(cls.__dict__.keys()):
            if attr.startswith("upos_xpos") or attr in ("some_property", "classname"):
                if hasattr(cls, attr):
                    delattr(cls, attr)
        # Also remove internal backing fields
        for obj in globals().values():
            if isinstance(obj, (Word, Sentence, Document)):
                for attr in list(obj.__dict__.keys()):
                    if attr.startswith("_"):
                        obj.__dict__.pop(attr, None)


class TestExtractReadonly:
    """BDD-style tests for extract_readonly."""

    # Given: A fresh Document class and input text
    # When: extract_readonly is called
    # Then: A read-only property is added and returned correctly
    def test_readonly_property_added_and_returned(self, nlp_pipeline):
        result = extract_readonly(
            input_text="Hello world!", property_name="greeting", property_value="hi", pipeline=nlp_pipeline
        )
        expected: Dict[str, Any] = {
            "input_text": "Hello world!",
            "greeting": "hi",
            "attempt_set": "(raises ValueError)",
        }
        assert result == expected
        assert hasattr(Document, "greeting")
        # Class-level is the property object; value is instance-level or via function result
        assert result["greeting"] == "hi"

    # Given: Property already exists
    # When: extract_readonly is called again
    # Then: It reuses existing property without error
    def test_idempotent_property_add(self, nlp_pipeline):
        extract_readonly(property_name="idempotent", property_value=42, pipeline=nlp_pipeline)
        result = extract_readonly(property_name="idempotent", property_value=999, pipeline=nlp_pipeline)
        expected: Dict[str, Any] = {
            "input_text": "This is a test document. Pretty cool!",
            "idempotent": 42,  # Value from first call
            "attempt_set": "(raises ValueError)",
        }
        assert result["idempotent"] == expected["idempotent"]


class TestExtractGetter:
    """BDD-style tests for extract_getter with upos_xpos_word format."""

    # Given: Input text with multiple sentences and words
    # When: extract_getter is called with default prop_name
    # Then: Returns tuples of "UPOS_XPOS_word" per sentence
    def test_default_getter_returns_upos_xpos_word_per_sentence(self, nlp_pipeline):
        input_text = "This is a test. It works well!"
        result = extract_getter(input_text=input_text, pipeline=nlp_pipeline)
        expected_tuples: Tuple[Tuple[str, ...], ...] = (
            ("PRON_DT_This", "AUX_VBZ_is", "DET_DT_a", "NOUN_NN_test", "PUNCT_._."),
            ("PRON_PRP_It", "VERB_VBZ_works", "ADV_RB_well", "PUNCT_._!"),
        )
        assert result["input_text"] == input_text
        assert result["upos_xpos_word_per_sentence"] == expected_tuples

    # Given: Custom property name and getter
    # When: Called with override
    # Then: Uses custom logic and name
    def test_custom_property_name_and_getter(self, nlp_pipeline):
        custom_getter = lambda self: f"{self.text.upper()}"
        result = extract_getter(
            input_text="Small case.",
            prop_name="uppercase",
            getter_fn=custom_getter,
            pipeline=nlp_pipeline,
        )
        # Stanza tokenises the trailing period as a separate Word with text "."
        expected_tuples: Tuple[Tuple[str, ...], ...] = (("SMALL", "CASE", "."),)
        assert result[f"{'uppercase'}_per_sentence"] == expected_tuples
        assert hasattr(Word, "uppercase")

    # Given: Property already defined
    # When: extract_getter called again
    # Then: No error, returns consistent values
    def test_getter_is_idempotent(self, nlp_pipeline):
        extract_getter(prop_name="upos_xpos_word", pipeline=nlp_pipeline)
        result = extract_getter(prop_name="upos_xpos_word", pipeline=nlp_pipeline)
        assert len(result["upos_xpos_word_per_sentence"]) > 0
        assert all(isinstance(s, tuple) for s in result["upos_xpos_word_per_sentence"])


class TestExtractSetterGetter:
    """BDD-style tests for extract_setter_getter."""

    # Given: Valid mapping and good value
    # When: Property is set via setter
    # Then: Internal value updates, getter returns string
    def test_setter_getter_with_valid_value(self, nlp_pipeline):
        result = extract_setter_getter(
            input_text="One sentence.",
            set_good_value="good",
            pipeline=nlp_pipeline,
        )
        expected = {
            "input_text": "One sentence.",
            "after_set_good": {"_classname": 1, "classname": "good"},
            "after_set_bad": {"_classname": 2, "classname": "bad"},
        }
        assert result == expected

    # Given: Custom mappings
    # When: Using non-default int2str/str2int
    # Then: Mapping is respected
    def test_custom_int2str_mapping(self, nlp_pipeline):
        int2str = {10: "alpha", 20: "beta"}
        str2int = {"alpha": 10, "beta": 20}
        result = extract_setter_getter(
            input_text="Test.",
            int2str=int2str,
            str2int=str2int,
            set_good_value="alpha",
            set_bad_internal=20,
            pipeline=nlp_pipeline,
        )
        expected = {
            "input_text": "Test.",
            "after_set_good": {"_classname": 10, "classname": "alpha"},
            "after_set_bad": {"_classname": 20, "classname": "beta"},
        }
        assert result == expected

    # Given: Setter called with invalid string
    # When: str2int lacks key
    # Then: KeyError in setter (not caught here — let it fail if misused)
    def test_setter_raises_key_error_on_invalid_string(self, nlp_pipeline):
        with pytest.raises(KeyError):
            extract_setter_getter(
                input_text="Fail.",
                set_good_value="unknown",
                pipeline=nlp_pipeline,
            )


class TestExtractBackpointer:
    """BDD-style tests for extract_backpointer."""

    # Given: Multi-sentence document with entity and tokens
    # When: extract_backpointer is called
    # Then: All backpointers resolve to correct sentence index
    def test_backpointers_point_to_correct_sentences(self, nlp_pipeline):
        input_text = "Chris Manning wrote a sentence. Then another."
        result = extract_backpointer(input_text=input_text, pipeline=nlp_pipeline)
        expected = {
            "input_text": input_text,
            "entity_sentence_index": 0,
            "first_word_sentence_index": 0,
            "last_token_sentence_index": 1,
        }
        assert result == expected

    # Given: Single token document
    # When: extract_backpointer called
    # Then: Backpointer is 0
    def test_single_token_document(self, nlp_pipeline):
        # Use short input without entities to avoid doc.ents[0] error
        result = extract_backpointer(input_text="Go.", pipeline=nlp_pipeline)
        assert result["first_word_sentence_index"] == 0
        assert result["last_token_sentence_index"] == 0