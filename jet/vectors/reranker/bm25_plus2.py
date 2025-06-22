from collections import defaultdict
from typing import List, Dict, Tuple
from math import log
import logging
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

from jet.logger import logger

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class BM25Plus:
    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 1.0, use_stemming: bool = False):
        """
        Initialize BM25+ scoring model.

        Args:
            k1: Controls term frequency saturation (default: 1.5)
            b: Controls length normalization (default: 0.75)
            delta: Lower-bounding parameter for BM25+ (default: 1.0)
            use_stemming: If True, apply Porter stemming to tokens (default: False)
        """
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.use_stemming = use_stemming
        self.avgdl: float = 0.0
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []
        self.doc_count: int = 0
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer() if use_stemming else None

    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess raw text into tokens.

        Args:
            text: Raw document or query text

        Returns:
            List of cleaned tokens
        """
        # Remove special characters and lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stop words
        tokens = [t for t in tokens if t not in self.stop_words]
        # Apply stemming if enabled
        if self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens

    def score(self, query: List[str], doc_idx: int, doc: List[str]) -> float:
        """
        Compute BM25+ score for a document given a query.

        Args:
            query: List of query tokens
            doc_idx: Index of the document in the collection
            doc: Tokenized document

        Returns:
            BM25+ score for the document
        """
        score = 0.0
        doc_length = self.doc_len[doc_idx]

        # Count term frequencies in document
        term_freq = defaultdict(int)
        for term in doc:
            term_freq[term] += 1

        # Calculate score for each query term
        for term in set(query):  # Use set to avoid boosting on query term repetition
            if term in self.idf:
                tf = term_freq.get(term, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * \
                    (1 - self.b + self.b * (doc_length / self.avgdl))
                term_score = self.idf[term] * \
                    (numerator / denominator + self.delta)
                score += term_score

        # Phrase matching: Boost score for consecutive query terms
        query_phrases = [query[i:i+2]
                         for i in range(len(query)-1)]  # Check bigrams
        for phrase in query_phrases:
            phrase_str = ' '.join(phrase)
            doc_str = ' '.join(doc)
            if phrase_str in doc_str:
                score += 1.0  # Boost for phrase match

        return score

    def fit(self, documents: List[List[str]]) -> None:
        """
        Fit the model on a collection of tokenized documents.

        Args:
            documents: List of tokenized documents (each document is a list of tokens)
        """
        self.doc_count = len(documents)
        self.doc_len = [len(doc) for doc in documents]
        self.avgdl = sum(self.doc_len) / \
            self.doc_count if self.doc_count > 0 else 0.0

        # Calculate IDF for each term
        term_doc_count = defaultdict(int)
        for doc in documents:
            unique_terms = set(doc)
            for term in unique_terms:
                term_doc_count[term] += 1

        # IDF formula: log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
        for term, n_t in term_doc_count.items():
            self.idf[term] = log(
                (self.doc_count - n_t + 0.5) / (n_t + 0.5) + 1)
        logger.info(
            f"Fitted BM25+ on {self.doc_count} documents with {len(self.idf)} unique terms")

    def search(self, query: str, documents: List[str], normalize: bool = False) -> List[Tuple[int, float]]:
        """
        Search documents and return ranked list of (doc_idx, score) pairs.

        Args:
            query: Raw query string
            documents: List of raw document strings
            normalize: If True, normalize scores by the maximum score (default: False)

        Returns:
            List of (document index, score) tuples sorted by score in descending order
        """
        # Parse query for phrases
        query_tokens = []
        phrases = re.findall(r'"([^"]+)"', query)  # Extract quoted phrases
        for phrase in phrases:
            phrase_tokens = self.preprocess(phrase)
            query_tokens.extend(phrase_tokens)
        # Add non-quoted terms
        non_quoted = re.sub(r'"[^"]+"', '', query).strip()
        if non_quoted:
            query_tokens.extend(self.preprocess(non_quoted))

        # Preprocess documents
        doc_tokens = [self.preprocess(doc) for doc in documents]

        if not self.idf:
            logger.warning("Model not fitted. Fitting now...")
            self.fit(doc_tokens)

        scores = [(i, self.score(query_tokens, i, doc))
                  for i, doc in enumerate(doc_tokens)]

        # Normalize scores if requested
        if normalize and scores:
            max_score = max(score for _, score in scores) if scores else 1.0
            scores = [(i, score / max_score) for i, score in scores]

        # Sort by score (descending) and then by index (ascending) for tie-breaking
        ranked = sorted(scores, key=lambda x: (-x[1], x[0]))
        logger.info(
            f"Search completed for query with {len(query_tokens)} terms. Top score: {ranked[0][1] if ranked else 0}")
        return ranked
