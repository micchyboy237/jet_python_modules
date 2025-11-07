from typing import Generator, Optional, TypedDict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
from jet.adapters.keybert.embeddings import KeyBERTLlamacppEmbedder
from jet.logger.timer import time_it
from jet.models.model_types import EmbedModelType
from jet.wordnet.sentence import split_by_punctuations, split_sentences
from jet.wordnet.words import get_words
from jet.adapters.keybert import KeyBERT
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import spacy
import nltk
import spacy.tokens

nltk.download('stopwords', quiet=True)
ENGLISH_STOPWORDS = set(stopwords.words('english'))


class PhraseGram(TypedDict):
    phrase: str
    score: float


class DetectedPhrase(TypedDict):
    index: int
    sentence: str
    phrases: list[str]
    results: list[PhraseGram]


class QueryPhraseResult(TypedDict):
    query: str
    phrase: str
    score: float


class PhraseDetector:
    phrasegrams: dict[str, float]
    _spacy_cache: dict[str, spacy.tokens.Doc]

    def __init__(self, sentences: list[str], model_path: Optional[str] = None, embed_model: EmbedModelType = "all-MiniLM-L6-v2",
                 min_count: int = 3, threshold: float = 0.1, punctuations_split: list[str] = [',', '/', ':'],
                 reset_cache: bool = False, *args, **kwargs):
        self.model_path = model_path
        self.punctuations_split = punctuations_split
        self.min_count = min_count
        self.threshold = threshold
        embedder = KeyBERTLlamacppEmbedder(embedding_model="embeddinggemma", use_cache=True)
        self.model = KeyBERT(embedder)
        self.st_model = embedder  # Store for semantic querying
        # Enable parser for noun_chunks
        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
        self._spacy_cache = {}  # Initialize spaCy cache

        # Use language-based stopwords
        self.stopwords = set(stopwords.words("english"))
        self.max_cache_size = kwargs.get("max_cache_size", 5000)

        # Parallelism configuration
        self.max_workers = kwargs.get("max_workers", 4)

        if not sentences or len(sentences) < 2:
            raise ValueError("'sentences' must have at least 2 items.")
        sentences = list(set(sentences))
        self.phrasegrams = self._build_phrasegrams(sentences)

        if self.model_path and (not os.path.exists(self.model_path) or reset_cache):
            self.save_model()
        elif self.model_path:
            self.load_model()

    def _filter_phrases(self, phrases: List[tuple[str, float]], sentence: str) -> List[tuple[str, float]]:
        # Use cached spaCy Doc if available, otherwise process and cache
        if sentence.lower() not in self._spacy_cache:
            self._spacy_cache[sentence.lower()] = self.nlp(sentence.lower())
        doc = self._spacy_cache[sentence.lower()]
        noun_phrases = {chunk.text.lower() for chunk in doc.noun_chunks}
        filtered = []
        for phrase, score in phrases:
            if phrase not in self._spacy_cache:
                self._spacy_cache[phrase] = self.nlp(phrase)
            doc_phrase = self._spacy_cache[phrase]
            is_valid = (
                phrase in noun_phrases and
                all(token.pos_ in ["NOUN", "PROPN", "ADJ", "ADV"] for token in doc_phrase) and
                score >= 0.5
            )
            if is_valid:
                filtered.append((phrase, score))

        # Prevent unbounded cache growth
        if len(self._spacy_cache) > self.max_cache_size:
            self._spacy_cache.clear()
        return filtered

    @time_it
    def _build_phrasegrams(self, sentences: List[str]) -> dict[str, float]:
        phrasegrams = {}
        # Pre-filter sentences with at least 2 words
        valid_sentences = [s for s in sentences if len(get_words(s)) >= 2]
        if not valid_sentences:
            return phrasegrams

        # Batch process sub-sentences
        all_sub_sentences = []
        sentence_map = []
        for sentence in valid_sentences:
            sub_sentences = split_by_punctuations(
                sentence, self.punctuations_split)
            for sub_sentence in sub_sentences:
                if len(get_words(sub_sentence)) >= 2:
                    all_sub_sentences.append(sub_sentence.lower())
                    sentence_map.append(sub_sentence)

        # Batch keyword extraction
        if all_sub_sentences:
            keyphrases_batch = self.model.extract_keywords(
                all_sub_sentences,
                keyphrase_ngram_range=(2, 2),
                stop_words=list(ENGLISH_STOPWORDS),
                top_n=self.min_count + 2,
                use_mmr=True
            )

        # Parallel filtering
        tasks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for sub_sentence, keyphrases, original_sentence in zip(all_sub_sentences, keyphrases_batch, sentence_map):
                tasks.append(executor.submit(self._filter_phrases, keyphrases, original_sentence))
            for sub_sentence, future in zip(all_sub_sentences, as_completed(tasks)):
                for phrase, score in future.result():
                    if score >= self.threshold:
                        phrasegrams[phrase] = max(phrasegrams.get(phrase, 0), score)

        return phrasegrams

    @time_it
    def load_model(self, model_path: Optional[str] = None, *args, **kwargs) -> None:
        path = model_path if model_path is not None else self.model_path
        if path is None:
            raise ValueError(
                "No model_path provided and self.model_path is None")
        try:
            self.phrasegrams = joblib.load(path)
        except Exception:
            raise IOError(f"Failed to load model from {path}. File may be corrupt.")
        self._spacy_cache = {}  # Reset cache on load

    @time_it
    def save_model(self, model_path: Optional[str] = None) -> None:
        path = model_path if model_path is not None else self.model_path
        if path is None:
            raise ValueError(
                "No model_path provided and self.model_path is None")
        joblib.dump(self.phrasegrams, path)

    @time_it
    def detect_phrases(self, texts: List[str]) -> Generator[DetectedPhrase, None, None]:
        iterable = tqdm(
            texts, desc="Detecting phrases in texts") if tqdm else texts
        for idx, text in enumerate(iterable):
            sentences = split_sentences(text)
            sentence_iterable = tqdm(
                sentences, desc=f"Processing text {idx+1}", leave=False) if tqdm else sentences
            sentence_dict = {}
            for sentence in sentence_iterable:
                sub_sentences = split_by_punctuations(
                    sentence, self.punctuations_split)
                for sub_sentence in sub_sentences:
                    keyphrases = self.model.extract_keywords(
                        sub_sentence.lower(),
                        keyphrase_ngram_range=(2, 2),
                        stop_words=list(ENGLISH_STOPWORDS),
                        top_n=self.min_count + 2,
                        use_mmr=False
                    )
                    filtered_phrases = self._filter_phrases(
                        keyphrases, sub_sentence)
                    for phrase, score in filtered_phrases:
                        if score >= self.threshold and phrase not in sentence_dict:
                            sentence_dict[phrase] = {
                                "phrase": phrase, "score": score}
                if sentence_dict:
                    sorted_items = sorted(
                        sentence_dict.items(),
                        key=lambda x: x[1]['score'],
                        reverse=True
                    )
                    yield {
                        "index": idx,
                        "sentence": sentence,
                        "phrases": [item[0] for item in sorted_items],
                        "results": [item[1] for item in sorted_items]
                    }
                sentence_dict.clear()

    @time_it
    def extract_phrases(self, texts: List[str]) -> List[str]:
        results: List[str] = []
        seen_phrases = set()
        iterable = tqdm(
            texts, desc="Extracting phrases from texts") if tqdm else texts
        for text in iterable:
            sentences = split_sentences(text.lower())
            sentence_iterable = tqdm(
                sentences, desc="Processing sentences", leave=False) if tqdm else sentences
            for sentence in sentence_iterable:
                sub_sentences = split_by_punctuations(
                    sentence, self.punctuations_split)
                for sub_sentence in sub_sentences:
                    keyphrases = self.model.extract_keywords(
                        sub_sentence.lower(),
                        keyphrase_ngram_range=(2, 2),
                        stop_words=list(ENGLISH_STOPWORDS),
                        top_n=self.min_count + 2,
                        use_mmr=False
                    )
                    filtered_phrases = self._filter_phrases(
                        keyphrases, sub_sentence)
                    for phrase, score in filtered_phrases:
                        if score >= self.threshold and phrase not in seen_phrases:
                            seen_phrases.add(phrase)
                            results.append(phrase)
        return sorted(results)

    def get_phrase_grams(self, threshold: Optional[float] = None) -> dict[str, float]:
        phrase_grams = self.phrasegrams
        sorted_phrase_grams = sorted(
            phrase_grams.items(), key=lambda x: x[1], reverse=True
        )
        results = [
            {"phrase": phrase, "score": score}
            for phrase, score in sorted_phrase_grams
            if threshold is None or score >= threshold
        ]
        return {result["phrase"]: result["score"] for result in results}

    @time_it
    def query(self, queries: str | List[str]) -> List[QueryPhraseResult]:
        if isinstance(queries, str):
            queries = [queries]
        queries = self.transform_queries(queries)
        phrase_grams = self.get_phrase_grams()
        results: List[QueryPhraseResult] = []
        for phrase, score in phrase_grams.items():
            for query in queries:
                query_clean = query.replace("_", " ")
                if query_clean in phrase:
                    results.append({
                        "query": query,
                        "phrase": phrase,
                        "score": score,
                    })
        return sorted(results, key=lambda x: x["phrase"])

    @time_it
    def semantic_query(
        self,
        queries: str | List[str],
        top_k: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[QueryPhraseResult]:
        """
        Performs semantic query matching using cosine similarity between
        query embeddings and phrasegram embeddings.

        Args:
            queries: String or list of query strings.
            top_k: Number of top results to return per query.
            similarity_threshold: Minimum cosine similarity threshold.

        Returns:
            List of QueryPhraseResult, sorted by similarity score descending.
        """
        if isinstance(queries, str):
            queries = [queries]
        # Lowercase queries & convert spaces to underscores
        processed_queries = [q.strip().lower() for q in queries]
        phrase_grams = self.get_phrase_grams()
        phrases = list(phrase_grams.keys())
        if not phrases or not processed_queries:
            return []

        # Embed queries and phrasegrams
        all_embeds = self.st_model.embed(processed_queries + phrases)
        query_embeds = all_embeds[:len(processed_queries)]
        phrase_embeds = all_embeds[len(processed_queries):]
        similarities = cosine_similarity(query_embeds, phrase_embeds)

        results: List[QueryPhraseResult] = []
        for i, query in enumerate(processed_queries):
            sim_scores = similarities[i]
            indices = np.argwhere(sim_scores >= similarity_threshold).flatten()
            # If not enough, get top_k most similar anyway
            if len(indices) < top_k:
                indices = np.argsort(sim_scores)[-top_k:]
            # Sort indices in descending similarity
            ranked = sorted(indices, key=lambda idx: sim_scores[idx], reverse=True)
            for idx in ranked[:top_k]:
                results.append({
                    "query": queries[i],  # return original query (un-lowercased)
                    "phrase": phrases[idx],
                    "score": sim_scores[idx].item() if hasattr(sim_scores[idx], 'item') else float(sim_scores[idx]),
                })
        return sorted(results, key=lambda x: x["score"], reverse=True)

    @time_it
    def transform_queries(self, queries: str | List[str]) -> List[str]:
        if isinstance(queries, str):
            queries = [queries]
        return [query.lower().replace(" ", "_") for query in queries]
