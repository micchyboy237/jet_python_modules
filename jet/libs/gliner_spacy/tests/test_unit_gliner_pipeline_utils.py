import pytest
from pathlib import Path
import spacy
from spacy.tokens import Doc

from jet.libs.gliner_spacy.gliner_pipeline_utils import (
    CategoryData,
    build_label_set,
    init_gliner_pipeline,
    load_text_from_file,
    process_text,
    extract_sentence_themes,
)


# ---------- Fixtures ----------

@pytest.fixture
def tmp_text_file(tmp_path: Path) -> Path:
    """Create a temporary text file for file loading tests."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is a test sentence.", encoding="utf-8")
    return file_path


@pytest.fixture
def sample_cat_data() -> CategoryData:
    """Provide example category data."""
    return {
        "family": ["child", "spouse", "family", "parent"],
        "labor": ["work", "job", "office"],
        "education": ["school", "student"],
        "movement": ["move", "walk"],
        "violence": ["attack", "fear"],
    }


@pytest.fixture
def dummy_nlp() -> spacy.Language:
    """Lightweight spaCy pipeline for testing process_text."""
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


@pytest.fixture
def dummy_doc(dummy_nlp: spacy.Language) -> Doc:
    """Return a small processed document for extraction tests."""
    text = "John is a student. He loves his family."
    return dummy_nlp(text)


# ---------- Unit Tests ----------

class TestBuildLabelSet:
    """Unit tests for label building."""

    def test_unique_labels(self, sample_cat_data: CategoryData):
        # Given: category data with possible duplicates
        result = build_label_set(sample_cat_data)

        # When: labels are flattened and deduplicated
        expected = sorted({
            "child", "spouse", "family", "parent",
            "work", "job", "office",
            "school", "student",
            "move", "walk",
            "attack", "fear",
        })

        # Then: should return sorted unique list
        assert result == expected


class TestLoadTextFromFile:
    """Unit tests for file handling."""

    def test_load_existing_file(self, tmp_text_file: Path):
        # Given: a valid file path
        result = load_text_from_file(tmp_text_file)

        # Then: contents match expected
        expected = "This is a test sentence."
        assert result == expected

    def test_load_missing_file_raises(self):
        # Given: nonexistent file path
        with pytest.raises(FileNotFoundError):
            load_text_from_file("does_not_exist.txt")


class TestInitPipeline:
    """Unit tests for initializing pipeline."""

    def test_pipeline_contains_required_components(self, sample_cat_data: CategoryData):
        # Given: valid category data
        nlp = init_gliner_pipeline(sample_cat_data)

        # When: pipeline initialized
        pipes = nlp.pipe_names

        # Then: must contain all expected components
        assert "sentencizer" in pipes
        assert "gliner_spacy" in pipes
        assert "gliner_cat" in pipes


class TestProcessText:
    """Unit tests for text processing."""

    def test_process_text_returns_doc(self, dummy_nlp: spacy.Language):
        # Given: a valid nlp and text
        text = "Hello world."

        # When: processed through nlp
        doc = process_text(dummy_nlp, text)

        # Then: output must be a spaCy Doc
        assert isinstance(doc, Doc)
        assert doc.text == text


class TestExtractSentenceThemes:
    """Unit tests for extracting sentence details."""

    def test_extract_valid_sentence(self, dummy_doc: Doc):
        # Given: a dummy Doc with multiple sentences
        result = extract_sentence_themes(dummy_doc, 0)

        # Then: must contain expected keys and types
        assert isinstance(result["text"], str)
        assert isinstance(result["spans"], list)
        assert "raw_scores" in result

    def test_invalid_sentence_index_raises(self, dummy_doc: Doc):
        # Given: invalid index
        with pytest.raises(IndexError):
            extract_sentence_themes(dummy_doc, 10)
