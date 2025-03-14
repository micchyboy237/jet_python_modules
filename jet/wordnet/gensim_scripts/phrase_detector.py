from typing import Optional, TypedDict
from jet.logger import time_it
from jet.search.transformers import clean_string
from jet.wordnet.sentence import split_by_punctuations, split_sentences
from jet.wordnet.words import get_words
from gensim.models.phrases import Phrases
from jet.wordnet.analyzers.helpers import (
    get_tl_corpus_sentences,
    get_en_translation_sentences,
)
# from instruction_generator.scripts.extract_corpus_tagalog import filter_sample_sentences
from tqdm import tqdm
import os

ENGLISH_CONNECTOR_WORDS = frozenset(
    " a an the "  # articles; we never care about these in MWEs
    # prepositions; incomplete on purpose, to minimize FNs
    " for of with without at from to in on by "
    " and or "  # conjunctions; incomplete on purpose, to minimize FNs
    .split()
)

TAGALOG_CONNECTOR_WORDS = frozenset(
    " at ng sa "  # prepositions; incomplete on purpose, to minimize FNs
    " at o "  # conjunctions; incomplete on purpose, to minimize FNs
    " ay "  # part
    " ang mga si "  # article
    " na nang "  # connector
    .split()
)

# Combine the connector words for English and Tagalog
CONNECTOR_WORDS = ENGLISH_CONNECTOR_WORDS.union(TAGALOG_CONNECTOR_WORDS)


class PhraseGram(TypedDict):
    phrase: str
    score: float


class QueryPhraseResult(TypedDict):
    query: str
    phrase: str
    score: float


class PhraseDetector:
    phrasegrams: dict[str, float]

    @time_it
    def __init__(self, model_path: str, sentences: list[str] = [], min_count=3, threshold: float = 0.1, *args, punctuations_split: list[str] = [',', '/', ':'], **kwargs):
        self.punctuations_split = punctuations_split

        if not os.path.exists(model_path):
            print("Preprocessing sentences")

            if not sentences or len(sentences) < 2:
                raise ValueError("'sentences' must have at least 2 items.")

            cleaned_sentences = [
                clean_string(sentence.lower())
                for sentence in sentences
            ]
            cleaned_sentences = list(set(cleaned_sentences))

            lower_words = []
            for sentence in cleaned_sentences:
                sub_sentences = split_by_punctuations(
                    sentence, self.punctuations_split)
                for sub_sentence in sub_sentences:
                    words = get_words(sub_sentence)
                    words = [word.lower() for word in words]
                    lower_words.append(words)
            print("Creating new model")
            self.model = Phrases(
                lower_words,
                *args,
                min_count=min_count,
                threshold=threshold,
                scoring='npmi',
                connector_words=CONNECTOR_WORDS,
                **kwargs)
            print(f"Saving model to {model_path}")
            self.save_model(model_path)

        print(f"Loading model from {model_path}")
        self.load_model(model_path, *args, **kwargs)

    def __getattr__(self, name):
        """
        Delegate attribute lookup to self.model.

        This method is called if the attribute `name` isn't found in the
        PhraseDetector instance. If `name` is a method or attribute of self.model,
        it returns that method/attribute. Otherwise, it raises an AttributeError.
        """
        # Check if the attribute exists in self.model and return it.
        # This allows direct access to methods and properties of self.model.
        try:
            return getattr(self.model, name)
        except AttributeError:
            # If the attribute is not found in self.model, raise an AttributeError
            # to signal that this object doesn't have the requested attribute.
            raise AttributeError(
                f"'Phrases' object has no attribute '{name}'")

    @time_it
    def load_model(self, model_path, *args, **kwargs) -> None:
        model = Phrases.load(model_path, *args, **kwargs)
        self.phrasegrams = model.phrasegrams
        self.model = model

    @time_it
    def save_model(self, model_path):
        frozen_model = self.model.freeze()
        frozen_model.save(model_path)

    @time_it
    def detect_phrases(self, texts: list[str]) -> list:
        phrases = []

        for idx, text in enumerate(texts):
            sentences = split_sentences(text)
            cleaned_sentences = [
                clean_string(sentence.lower())
                for sentence in sentences
            ]
            cleaned_sentences = list(set(cleaned_sentences))

            for sentence in cleaned_sentences:
                sentence_dict = {}
                sub_sentences = split_by_punctuations(
                    sentence, self.punctuations_split)
                for sub_sentence in sub_sentences:
                    for phrase, score in self.model.analyze_sentence(get_words(sub_sentence)):
                        if score:
                            obj = {
                                "phrase": phrase,
                                "score": score
                            }
                            sentence_dict[phrase] = obj

                if sentence_dict:
                    # Sort by score in descending order
                    sorted_items = sorted(sentence_dict.items(
                    ), key=lambda x: x[1]['score'], reverse=True)

                    yield {
                        "index": idx,
                        "sentence": sentence,
                        # Sorted phrases
                        "phrases": [item[0] for item in sorted_items],
                        # Sorted results
                        "results": [item[1] for item in sorted_items]
                    }

        return phrases

    @time_it
    def get_phrase_grams(self, threshold: Optional[float] = None) -> dict[str, float]:
        phrase_grams: dict[str, float] = self.phrasegrams

        # Sort by values (scores) in descending order
        sorted_phrase_grams = sorted(
            phrase_grams.items(), key=lambda x: x[1], reverse=True)

        # Convert to list of TypedDicts
        results: list[PhraseGram] = [
            {"phrase": phrase, "score": score} for phrase, score in sorted_phrase_grams]

        # Apply threshold if provided
        if threshold is not None:
            results = [
                entry for entry in results if entry["score"] >= threshold]

        return {result["phrase"]: result["score"] for result in results}

    @time_it
    def query(self, queries: str | list[str]) -> list[QueryPhraseResult]:
        if isinstance(queries, str):
            queries = [queries]

        # Lowercase all queries
        queries = [query.lower() for query in queries]

        phrase_grams = self.get_phrase_grams()

        results: list[QueryPhraseResult] = []
        for phrase, score in phrase_grams.items():
            for query in queries:
                if query in phrase:
                    results.append({
                        "query": query,
                        "phrase": phrase,
                        "score": score,
                    })
        return results


def detect_common_phrase_sentences(
    sentences: list[str],
    max_phrase_duplicates: int = 2,
    min_coverage: float = None,
) -> list[str]:
    model_path = 'instruction_generator/wordnet/embeddings/gensim_jet_phrase_model.pkl'

    detector = PhraseDetector(model_path, sentences)
    phrases_stream = detector.detect_phrases(sentences)
    results_dict = {}
    added_sentences = set()
    phrase_counts = {}  # Track the number of occurrences of each phrase

    for item in tqdm(list(phrases_stream), desc="Detecting phrases"):
        phrases = item["phrases"]
        sentence = item["sentence"]
        results = item["results"]

        if isinstance(max_phrase_duplicates, int):
            # Check if at least one phrase exceeds the max allowed duplicates
            if any(phrase_counts.get(phrase, 0) > max_phrase_duplicates for phrase in phrases):
                continue

            # Update phrase counts
            for phrase in phrases:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        if sentence in added_sentences:
            continue

        # Calculate coverage
        sentence_length = len(sentence.split())
        phrase_lengths = sum(len(phrase.split()) for phrase in phrases)

        coverage = phrase_lengths / sentence_length if sentence_length else 0
        average_score = sum(result["score"]
                            for result in results) / len(results)

        if isinstance(min_coverage, float):
            if coverage < min_coverage:
                continue

        obj = {
            "phrases": phrases,
            "sentence": sentence,
            "coverage": coverage,
            "average_score": average_score,
            "results": results
        }

        results_dict[sentence] = obj
        added_sentences.add(sentence)

    # Get all sentences
    results = list(results_dict.values())
    # Reverse sort results by coverage
    results = sorted(results, key=lambda x: x["coverage"], reverse=True)

    return results


if __name__ == '__main__':
    model_path = 'instruction_generator/wordnet/embeddings/gensim_jet_phrase_model.pkl'
    # en_sentences = get_en_translation_sentences()
    tl_sentences = get_tl_corpus_sentences() if not os.path.exists(model_path) else []
    # tl_sentences = filter_sample_sentences(
    #     tl_sentences, case_sensitive=False, lang='tl')

    sentences = tl_sentences
    sentences = [sentence.lower() for sentence in sentences]
    sentences = list(set(sentences))
    print(f"Number of sentences: {len(sentences)}")

    detector = PhraseDetector(model_path, sentences)

    sentences_for_analysis = [
        "Ang basketbol ay magandang laro",
        "Kailangan ko nang matulog.",
    ]
    phrases_stream = detector.detect_phrases(sentences_for_analysis)

    for item in phrases_stream:
        sentence = item["sentence"]
        phrases = item["phrases"]
        results = item["results"]

        print(
            f"Sentence: {sentence}\nPhrases: {phrases}\nResults: {results}\n")

    # phrase_grams = detector.get_phrase_grams()
    # print(f"Phrase grams: {len(phrase_grams)}")
