from typing import List, Dict, Tuple, TypedDict
from textblob import TextBlob
from textblob.tokenizers import SentenceTokenizer, WordTokenizer
from textblob.en.np_extractors import ConllExtractor, FastNPExtractor
from textblob.en.taggers import NLTKTagger
from textblob.en.sentiments import PatternAnalyzer
import logging
import nltk

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Chunk(TypedDict):
    """Type definition for a processed text chunk."""
    text: str
    noun_phrases: List[str]
    pos_tags: List[Tuple[str, str]]
    sentiment: Dict[str, float]


class RAGPreprocessor:
    """Preprocesses text for RAG embeddings search using TextBlob."""

    def __init__(
        self,
        tokenizer=SentenceTokenizer(),
        np_extractor=ConllExtractor(),
        pos_tagger=NLTKTagger(),  # Switched to NLTKTagger for better POS accuracy
        analyzer=PatternAnalyzer()
    ):
        self.tokenizer = tokenizer
        self.np_extractor = np_extractor
        self.pos_tagger = pos_tagger
        self.analyzer = analyzer
        self.word_tokenizer = WordTokenizer()

    def create_blob(self, text: str, word_tokenize: bool = False) -> TextBlob:
        """Creates a TextBlob instance with configured settings."""
        tokenizer = self.word_tokenizer if word_tokenize else self.tokenizer
        # Ensure punctuation is included by tokenizing with nltk.word_tokenize
        if word_tokenize:
            tokens = nltk.word_tokenize(text)
            # Rejoin tokens to ensure proper formatting
            text = " ".join(tokens)
        return TextBlob(
            text,
            tokenizer=tokenizer,
            np_extractor=self.np_extractor,
            pos_tagger=self.pos_tagger,
            analyzer=self.analyzer
        )

    def preprocess_for_rag(self, document: str) -> List[Chunk]:
        """
        Preprocesses a document into chunks for RAG embeddings search.

        Args:
            document: Raw text document to process.

        Returns:
            List of dictionaries containing text chunks and metadata.
        """
        blob = self.create_blob(
            document)  # Use SentenceTokenizer for splitting
        chunks: List[Chunk] = []

        for sentence in blob.sentences:
            sentence_text = str(sentence)
            # Use WordTokenizer and NLTKTagger for sentence-level processing
            sentence_blob = self.create_blob(sentence_text, word_tokenize=True)

            # Debug logging
            logger.debug(f"Processing sentence: {sentence_text}")
            logger.debug(f"POS tags: {sentence_blob.tags}")
            logger.debug(f"Raw noun phrases: {sentence_blob.noun_phrases}")

            chunk: Chunk = {
                "text": sentence_text,
                "noun_phrases": sentence_blob.noun_phrases,
                "pos_tags": sentence_blob.tags,
                "sentiment": {
                    "polarity": sentence_blob.sentiment.polarity,
                    "subjectivity": sentence_blob.sentiment.subjectivity
                }
            }
            chunks.append(chunk)

        return chunks
