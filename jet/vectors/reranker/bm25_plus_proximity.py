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

    def score(self, query: List[str], doc_idx: int, doc: List[str], phrases: List[List[str]]) -> Tuple[float, int]:
        """
        Compute BM25+ score for a document given a query.

        Args:
            query: List of query tokens
            doc_idx: Index of the document in the collection
            doc: Tokenized document
            phrases: List of phrase token lists to match

        Returns:
            Tuple of (BM25+ score for the document, number of matched terms)
        """
        score = 0.0
        doc_length = self.doc_len[doc_idx]

        # Count term frequencies in document
        term_freq = defaultdict(int)
        for term in doc:
            term_freq[term] += 1

        # Calculate score for each query term
        matched_terms = 0
        non_phrase_terms = set(
            query) - set(term for phrase in phrases for term in phrase)
        term_scores = {}
        for term in set(query):
            if term in self.idf:
                tf = term_freq.get(term, 0)
                matched_terms += 1 if tf > 0 else 0
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * \
                    (1 - self.b + self.b * (doc_length / self.avgdl))
                term_score = self.idf[term] * \
                    (numerator / denominator + self.delta)
                score += term_score
                term_scores[term] = term_score
        logger.debug(
            f"Doc {doc_idx} term scores: {term_scores}, doc_length: {doc_length}, avgdl: {self.avgdl}")

        # Phrase matching: Boost score for exact phrase matches
        phrase_positions = []
        phrase_boost_total = 0.0
        for phrase in phrases:
            phrase_len = len(phrase)
            if phrase_len < 1:
                continue
            phrase_count = 0
            for i in range(len(doc) - phrase_len + 1):
                if doc[i:i+phrase_len] == phrase:
                    phrase_count += 1
                    phrase_positions.append(i)
            if phrase_count > 0:
                phrase_boost = phrase_len * phrase_count * 2.5  # Increased from 2.0 to 2.5
                phrase_boost_total += phrase_boost
                score += phrase_boost
        logger.debug(
            f"Doc {doc_idx} phrase boost: {phrase_boost_total}, phrase_positions: {phrase_positions}")

        # Proximity bonus for non-phrase terms near phrases, considering sentence boundaries
        proximity_boost_total = 0.0
        if phrase_positions and non_phrase_terms:
            sentence_breaks = [i for i, token in enumerate(doc) if token in [
                '.', '!', '?']]
            sentence_breaks = [-1] + sentence_breaks + [len(doc)]
            term_positions = defaultdict(list)
            for term in non_phrase_terms:
                if term in term_freq:
                    for i, token in enumerate(doc):
                        if token == term:
                            term_positions[term].append(i)

            for term in non_phrase_terms:
                if term in term_positions:
                    term_proximity_boost = 0.0
                    for term_pos in term_positions[term]:
                        term_sentence_idx = next(i for i, pos in enumerate(
                            sentence_breaks) if pos >= term_pos)
                        min_proximity_boost = float('inf')
                        for phrase_pos in phrase_positions:
                            phrase_sentence_idx = next(i for i, pos in enumerate(
                                sentence_breaks) if pos >= phrase_pos)
                            distance = abs(term_pos - phrase_pos)
                            if term_sentence_idx == phrase_sentence_idx:
                                # Increased from 1.0 to 1.5
                                proximity_boost = 1.5 / (distance + 1)
                            else:
                                proximity_boost = 0.25 / (distance + 1)
                            min_proximity_boost = min(
                                min_proximity_boost, proximity_boost)
                        term_proximity_boost += min_proximity_boost
                    proximity_boost_total += term_proximity_boost
                    score += term_proximity_boost
        logger.debug(
            f"Doc {doc_idx} proximity boost: {proximity_boost_total}, non_phrase_terms: {non_phrase_terms}")

        logger.debug(
            f"Doc {doc_idx} final score: {score}, matched_terms: {matched_terms}")
        return score, matched_terms

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
        # Parse query for phrases and individual terms
        query_tokens = []
        phrases = []  # Store phrases as lists of tokens
        raw_phrases = re.findall(r'"([^"]+)"', query)  # Extract quoted phrases
        for phrase in raw_phrases:
            phrase_tokens = self.preprocess(phrase)
            if phrase_tokens:  # Only add non-empty phrases
                phrases.append(phrase_tokens)
                query_tokens.extend(phrase_tokens)
        # Add non-quoted terms
        non_quoted = re.sub(r'"[^"]+"', '', query).strip()
        if non_quoted:
            query_tokens.extend(self.preprocess(non_quoted))
        # Debug log
        logger.debug(f"Query tokens: {query_tokens}, phrases: {phrases}")

        # Preprocess documents
        doc_tokens = [self.preprocess(doc) for doc in documents]

        if not self.idf:
            logger.warning("Model not fitted. Fitting now...")
            self.fit(doc_tokens)

        # Score documents, including matched term count for tie-breaking
        scores = [(i, *self.score(query_tokens, i, doc, phrases))
                  for i, doc in enumerate(doc_tokens)]
        # Debug log
        logger.debug(f"Document scores with matched terms: {scores}")

        # Normalize scores if requested
        if normalize and scores:
            max_score = max(score for _, score, _ in scores) if scores else 1.0
            scores = [(i, score / max_score, matched_terms)
                      for i, score, matched_terms in scores]

        # Sort by score (descending), then matched terms (descending), then index (ascending)
        ranked = sorted(scores, key=lambda x: (-x[1], -x[2], x[0]))
        logger.info(
            f"Search completed for query with {len(query_tokens)} terms. Top score: {ranked[0][1] if ranked else 0}")
        # Return (index, score) tuples
        return [(i, score) for i, score, _ in ranked]
