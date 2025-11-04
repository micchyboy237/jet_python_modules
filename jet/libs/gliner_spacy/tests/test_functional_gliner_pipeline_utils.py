import pytest
import spacy
from spacy.tokens import Doc
from jet.libs.gliner_spacy.gliner_pipeline_utils import (
    CategoryData,
    process_text,
    visualize_doc,
)


@pytest.mark.integration
class TestGlinerPipelineFunctional:
    """Functional / integration tests for pipeline utilities."""

    def setup_method(self):
        # Given: sample category data and a simple spaCy pipeline
        self.cat_data: CategoryData = {
            "family": ["child", "parent"],
            "education": ["school"],
        }
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

    # ---------- Functional Tests ----------

    def test_end_to_end_processing(self):
        # Given: a short text with two themes
        text = "My father works in a school."

        # When: processing through spaCy pipeline
        doc = process_text(self.nlp, text)

        # Then: should return a valid Doc with sentences and tokens
        assert isinstance(doc, Doc)
        assert len(list(doc.sents)) == 1
        assert all(token.text for token in doc)

    def test_visualization_fails_without_extensions(self):
        # Given: a basic Doc without GliNER extensions
        doc = self.nlp("Sample sentence.")

        # When: calling visualize_doc
        # Then: raises AttributeError since no ._.visualize exists
        with pytest.raises(AttributeError):
            visualize_doc(doc)
