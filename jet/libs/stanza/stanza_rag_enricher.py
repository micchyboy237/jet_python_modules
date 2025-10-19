from typing import TypedDict, List
import stanza

class EnrichedToken(TypedDict):
    text: str
    lemma: str
    upos: str
    deprel: str
    head_text: str

class EnrichedDocument(TypedDict):
    original_text: str
    annotated_text: str  # e.g., "Word [deprel:POS] ..."
    tokens: List[EnrichedToken]
    sentences: List[str]  # Original sentences for chunking

class StanzaRAGEnricher:
    """
    Modular class to enrich documents with Stanza syntax features for LLM RAG.
    Focuses on POS, lemmatization, and dependency parsing to add structural context.
    Reusable for batch processing; configurable for languages/processors.
    """
    def __init__(self, lang: str = "en", processors: str = "tokenize,pos,lemma,depparse", use_gpu: bool = False):
        """
        Initialize Stanza pipeline.
        
        :param lang: Language code (e.g., 'en' for English).
        :param processors: Comma-separated Stanza processors (add 'constituency' for 2025 updates).
        :param use_gpu: Enable GPU for M1 or GTX 1660 acceleration.
        """
        self.nlp = stanza.Pipeline(lang=lang, processors=processors, use_gpu=use_gpu)

    def enrich_document(self, text: str) -> EnrichedDocument:
        """
        Process text and return enriched features for RAG context.
        
        :param text: Raw input text.
        :return: Typed dict with annotations for LLM prompt injection.
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sentences]
        tokens: List[EnrichedToken] = []
        annotated_parts = []

        for sent in doc.sentences:
            for word in sent.words:
                token = EnrichedToken(
                    text=word.text,
                    lemma=word.lemma,
                    upos=word.upos,
                    deprel=word.deprel,
                    head_text=next((w.text for w in sent.words if w.id == word.head), "")
                )
                tokens.append(token)
                annotated_parts.append(f"{word.text} [{word.deprel}:{word.upos}]")

            annotated_parts.append(" | ")  # Sentence boundary

        annotated_text = " ".join(annotated_parts).rstrip(" | ")
        return EnrichedDocument(
            original_text=text,
            annotated_text=annotated_text,
            tokens=tokens,
            sentences=sentences
        )