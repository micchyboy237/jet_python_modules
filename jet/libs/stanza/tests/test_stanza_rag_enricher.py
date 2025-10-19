import pytest
from jet.libs.stanza.stanza_rag_enricher import StanzaRAGEnricher, EnrichedDocument, EnrichedToken

@pytest.fixture
def enricher() -> StanzaRAGEnricher:
    """Fixture to initialize English pipeline without GPU for test portability."""
    return StanzaRAGEnricher(lang="en", use_gpu=False)

class TestStanzaRAGEnricher:
    def test_enrich_document_basic_syntax(self, enricher):
        # Given: A simple English sentence with known dependencies (from Stanza examples)
        input_text = "Barack Obama was born in Hawaii."
        expected_sentences = ["Barack Obama was born in Hawaii."]
        expected_annotated = (
            "Barack [nsubj:pass:PROPN] Obama [flat:PROPN] was [aux:pass:AUX] born [root:VERB] "
            "in [case:ADP] Hawaii [obl:PROPN] . [punct:PUNCT]"
        )
        expected_token = EnrichedToken(
            text="Obama", lemma="Obama", upos="PROPN", deprel="flat", head_text="Barack"
        )

        # When: Enrich the document
        result: EnrichedDocument = enricher.enrich_document(input_text)

        # Then: Verify structure, annotations, and exact token details
        assert result["original_text"] == input_text
        assert result["sentences"] == expected_sentences
        assert result["annotated_text"] == expected_annotated  # Exact match updated for passive parse
        assert any(t == expected_token for t in result["tokens"])  # Updated expected token for flat relation

    def test_enrich_document_multi_sentence(self, enricher):
        # Given: Multi-sentence text for RAG chunking
        input_text = "Iraqi authorities announced that they had busted up 3 terrorist cells. Two of them were run by officials."
        expected_sentences = [
            "Iraqi authorities announced that they had busted up 3 terrorist cells.",
            "Two of them were run by officials."
        ]

        # When: Enrich the document
        result: EnrichedDocument = enricher.enrich_document(input_text)

        # Then: Verify sentence splitting and token count
        assert result["sentences"] == expected_sentences
        assert len(result["tokens"]) == 20  # Updated to match actual tokenization (12 + 8 including punctuation)

    def test_enrich_document_lemma_normalization(self, enricher):
        # Given: Text with inflected forms for lemmatization check
        input_text = "The authorities announced that they had busted up cells operating in Baghdad."
        expected_lemmas = ["authority", "announce", "bust", "cell", "operate"]  # Key normalized forms

        # When: Enrich the document
        result: EnrichedDocument = enricher.enrich_document(input_text)

        # Then: Verify lemmas for RAG normalization (e.g., query matching "bust" to "busted")
        extracted_lemmas = [t["lemma"] for t in result["tokens"] if t["lemma"] in expected_lemmas]
        assert extracted_lemmas == expected_lemmas  # Exact list match

    def test_enrich_document_custom_processors(self):
        # Given: Enricher with constituency (2025 update) for advanced syntax
        enricher_const = StanzaRAGEnricher(lang="en", processors="tokenize,pos,lemma,depparse,constituency", use_gpu=False)
        input_text = "Barack Obama was born."

        # When: Enrich (constituency adds tree structure, but we focus on core output)
        result = enricher_const.enrich_document(input_text)

        # Then: Core features preserved; constituency accessible via doc (not in TypedDict for simplicity)
        assert len(result["sentences"]) == 1
        assert "[root:VERB]" in result["annotated_text"]  # Updated to check for 'born' as root in passive
