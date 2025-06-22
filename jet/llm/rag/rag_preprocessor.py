# jet/llm/rag/rag_preprocessor.py
from typing import List, Optional
import trafilatura
import logging
import spacy
from spacy.language import Language
import textacy.preprocessing as tprep

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WebDataPreprocessor:
    """Preprocesses web-scraped data for RAG usage."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """Initialize with chunking parameters and load Spacy model."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=[
                                  "ner", "lemmatizer"])
        except OSError:
            logger.error(
                "Spacy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            raise

    def fetch_and_extract(self, url: str) -> Optional[str]:
        """Fetch webpage and extract main content using trafilatura."""
        logger.info(f"Fetching and extracting content from {url}")
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                logger.warning(f"Failed to fetch content from {url}")
                return None
            extracted = trafilatura.extract(
                downloaded, include_comments=False, include_tables=False)
            if not extracted:
                logger.warning(f"No content extracted from {url}")
                return None
            logger.info(f"Successfully extracted content from {url}")
            return extracted
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return None

    def clean_text(self, text: str) -> str:
        """Clean extracted text for RAG processing using textacy."""
        logger.info("Cleaning text")
        cleaned = tprep.normalize.unicode(text, form="NFKC")
        cleaned = tprep.normalize.whitespace(cleaned)
        # Replace URLs with empty string
        cleaned = tprep.replace.urls(cleaned, "")
        # Replace emails with empty string
        cleaned = tprep.replace.emails(cleaned, "")
        cleaned = tprep.replace.phone_numbers(
            cleaned, "")  # Replace phone numbers
        cleaned = tprep.remove.punctuation(cleaned)  # Remove all punctuation
        # Remove excessive whitespace after textacy cleaning
        cleaned = " ".join(cleaned.split())
        logger.info("Text cleaning completed")
        return cleaned

    def chunk_text(self, text: str) -> List[str]:
        """Chunk text into semantically coherent pieces using Spacy."""
        logger.info("Chunking text")
        # Use Spacy for sentence boundary detection
        doc: Language = self.nlp(text)
        sentences = [sent.text.strip()
                     for sent in doc.sents if len(sent.text.strip()) > 10]
        if not sentences:
            logger.warning("No valid sentences found for chunking")
            return []

        # Join sentences and split into chunks
        text_to_chunk = " ".join(sentences)
        chunks = []
        start_idx = 0
        text_length = len(text_to_chunk)

        while start_idx < text_length:
            end_idx = min(start_idx + self.chunk_size, text_length)
            # Adjust end_idx to avoid splitting mid-sentence
            if end_idx < text_length:
                # Find the nearest sentence boundary before or at end_idx
                sub_doc = self.nlp(text_to_chunk[start_idx:end_idx])
                last_sent = list(sub_doc.sents)[-1] if sub_doc.sents else None
                if last_sent:
                    sentence_end = start_idx + last_sent.end_char
                    end_idx = min(sentence_end, end_idx)
            chunk = text_to_chunk[start_idx:end_idx].strip()
            if len(chunk) >= 10:  # Lowered threshold to allow smaller chunks
                chunks.append(chunk)
            start_idx = max(start_idx + self.chunk_size -
                            self.chunk_overlap, end_idx)

        logger.info(f"Generated {len(chunks)} valid chunks")
        return chunks
