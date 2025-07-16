from typing import Generator, Optional, TypedDict, List
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType
from jet.wordnet.sentence import split_by_punctuations, split_sentences
from jet.wordnet.words import get_words
from keybert import KeyBERT
from nltk.corpus import stopwords
import pickle
import os
import spacy
import nltk

# Initialize NLTK data
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

    def __init__(self, model_path: str, embed_model: EmbedModelType = "all-MiniLM-L6-v2", sentences: list[str] = [], min_count: int = 3, threshold: float = 0.1, *args, punctuations_split: list[str] = [',', '/', ':'], reset_cache: bool = False, **kwargs):
        self.punctuations_split = punctuations_split
        self.min_count = min_count
        self.threshold = threshold
        st_model = SentenceTransformerRegistry.load_model(embed_model)
        self.model = KeyBERT(st_model)
        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])

        if not os.path.exists(model_path) or reset_cache:
            if not sentences or len(sentences) < 2:
                raise ValueError("'sentences' must have at least 2 items.")
            sentences = list(set(sentences))
            self.phrasegrams = self._build_phrasegrams(sentences)
            self.save_model(model_path)
        else:
            self.load_model(model_path)

    def _filter_phrases(self, phrases: List[tuple[str, float]], sentence: str) -> List[tuple[str, float]]:
        doc = self.nlp(sentence.lower())
        noun_phrases = {chunk.text.lower() for chunk in doc.noun_chunks}
        filtered = []
        for phrase, score in phrases:
            doc_phrase = self.nlp(phrase)
            is_valid = (
                phrase in noun_phrases and
                all(token.pos_ in ["NOUN", "PROPN", "ADJ", "ADV"] for token in doc_phrase) and
                score >= 0.5
            )
            if is_valid:
                filtered.append((phrase, score))
        return filtered

    def _build_phrasegrams(self, sentences: List[str]) -> dict[str, float]:
        phrasegrams = {}
        for sentence in sentences:
            sub_sentences = split_by_punctuations(
                sentence, self.punctuations_split)
            for sub_sentence in sub_sentences:
                words = get_words(sub_sentence)
                if not words:
                    continue
                keyphrases = self.model.extract_keywords(
                    sub_sentence.lower(),
                    keyphrase_ngram_range=(2, 2),
                    stop_words=list(ENGLISH_STOPWORDS),
                    top_n=self.min_count + 2,
                    use_mmr=True
                )
                filtered_phrases = self._filter_phrases(
                    keyphrases, sub_sentence)
                for phrase, score in filtered_phrases:
                    if score >= self.threshold:
                        phrasegrams[phrase] = max(
                            phrasegrams.get(phrase, 0), score)
        return phrasegrams

    def load_model(self, model_path: str, *args, **kwargs) -> None:
        with open(model_path, 'rb') as f:
            self.phrasegrams = pickle.load(f)

    def save_model(self, model_path: str) -> None:
        with open(model_path, 'wb') as f:
            pickle.dump(self.phrasegrams, f)

    def detect_phrases(self, texts: List[str]) -> Generator[DetectedPhrase, None, None]:
        for idx, text in enumerate(texts):
            sentences = split_sentences(text)
            sentence_dict = {}
            for sentence in sentences:
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

    def extract_phrases(self, texts: List[str]) -> List[str]:
        results: List[str] = []
        seen_phrases = set()
        for text in texts:
            sentences = split_sentences(text.lower())
            for sentence in sentences:
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

    def transform_queries(self, queries: str | List[str]) -> List[str]:
        if isinstance(queries, str):
            queries = [queries]
        return [query.lower().replace(" ", "_") for query in queries]
