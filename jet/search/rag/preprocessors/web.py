from typing import List, Optional
from jet.logger import logger
import trafilatura
import spacy
from spacy.language import Language
import textacy.preprocessing as tprep


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
        cleaned = tprep.replace.urls(cleaned, "")
        cleaned = tprep.replace.emails(cleaned, "")
        cleaned = tprep.replace.phone_numbers(cleaned, "")
        cleaned = tprep.remove.punctuation(cleaned)
        cleaned = " ".join(cleaned.split())
        logger.info("Text cleaning completed")
        return cleaned

    def chunk_text(self, text: str) -> List[str]:
        """Chunk text into semantically coherent pieces using Spacy."""
        logger.info("Chunking text")
        doc: Language = self.nlp(text)
        sentences = [sent.text.strip()
                     for sent in doc.sents if len(sent.text.strip()) > 10]
        if not sentences:
            logger.warning("No valid sentences found for chunking")
            return []
        text_to_chunk = " ".join(sentences)
        chunks = []
        start_idx = 0
        text_length = len(text_to_chunk)
        while start_idx < text_length:
            end_idx = min(start_idx + self.chunk_size, text_length)
            if end_idx < text_length:
                sub_doc = self.nlp(text_to_chunk[start_idx:end_idx])
                last_sent = list(sub_doc.sents)[-1] if sub_doc.sents else None
                if last_sent:
                    sentence_end = start_idx + last_sent.end_char
                    end_idx = min(sentence_end, end_idx)
            chunk = text_to_chunk[start_idx:end_idx].strip()
            if len(chunk) >= 10:
                chunks.append(chunk)
            start_idx = max(start_idx + self.chunk_size -
                            self.chunk_overlap, end_idx)
        logger.info(f"Generated {len(chunks)} valid chunks")
        return chunks

    def preprocess(self, url: str) -> List[str]:
        """Full preprocessing pipeline: fetch, clean, and chunk."""
        logger.info(f"Starting preprocessing for {url}")
        raw_text = self.fetch_and_extract(url)
        if not raw_text:
            logger.error(f"Preprocessing failed: No content from {url}")
            return []
        cleaned_text = self.clean_text(raw_text)
        if not cleaned_text:
            logger.warning(f"Preprocessing resulted in empty text for {url}")
            return []
        chunks = self.chunk_text(cleaned_text)
        logger.info(
            f"Preprocessing completed for {url} with {len(chunks)} chunks")
        return chunks
