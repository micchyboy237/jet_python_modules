import math
import os
import re
from typing import Dict, List, Optional, TypedDict
from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker
from tqdm import tqdm
from jet.file.utils import load_data, save_data
from jet.logger import logger
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType
from jet.utils.string_utils import is_numeric
from jet.wordnet.pos_tagger_light import POSTagger
from jet.wordnet.words import get_words


class WeightsConfig(TypedDict):
    context: float
    word_similarity: float
    frequency: float


class SpellingCorrector:
    def __init__(self, data=None, embed_model: EmbedModelType = "mxbai-embed-large", dictionary_file=None, case_sensitive=False, ignore_words=None, base_words=None, count_threshold=5, weights: WeightsConfig = None):
        self.spell_checker = SpellChecker()
        self.tagger = POSTagger()
        self.case_sensitive = case_sensitive
        self.base_words = base_words or list(
            self.spell_checker.word_frequency.words())
        self.ignore_words = ignore_words + \
            self.base_words if ignore_words else self.base_words
        self.count_threshold = count_threshold
        self.unknown_words = set()  # Initialize unknown_words as an empty set
        self.sentence_model = SentenceTransformerRegistry.load_model(
            embed_model, device='mps')

        # Default weights
        default_weights = {
            "context": 0.35,
            "word_similarity": 0.0,
            "frequency": 0.65
        }
        self.weights = weights or default_weights

        # Validate weights
        required_keys = {"context", "word_similarity", "frequency"}
        if not isinstance(self.weights, dict):
            raise ValueError("Weights must be a dictionary")
        if set(self.weights.keys()) != required_keys:
            raise ValueError(
                f"Weights dictionary must contain exactly these keys: {required_keys}")
        for key, value in self.weights.items():
            if not isinstance(value, (int, float)) or value < 0.0:
                raise ValueError(
                    f"Weight for '{key}' must be a non-negative number, got {value}")
        weight_sum = sum(self.weights.values())
        if not math.isclose(weight_sum, 1.0, rel_tol=1e-5):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

        logger.info(f"Base words: {len(self.base_words)}")
        logger.info(f"Ignore words: {len(self.ignore_words)}")
        logger.info(f"Weights configuration: {self.weights}")
        logger.debug(f"Initialized unknown_words: {self.unknown_words}")
        if not data and dictionary_file and os.path.exists(dictionary_file):
            logger.info(f"Loading dictionary from {dictionary_file}")
            self.spell_checker.word_frequency.load_json(
                load_data(dictionary_file, is_binary=True))
        self.spell_checker.word_frequency.remove_by_threshold(
            self.count_threshold)

    def load_data_and_update_spellchecker(self, data):
        word_data = []
        for sentence in tqdm(data, desc="Loading data"):
            words = self.split_words(sentence)
            word_data.extend(words)
        self.spell_checker.word_frequency.load_words(word_data)

    def save_dictionary(self, dictionary_file):
        """Save the spell checker dictionary to a file."""
        save_data(dictionary_file, self.spell_checker.word_frequency.dictionary,
                  overwrite=True, is_binary=True)

    @staticmethod
    def split_compound_word(word, spell_checker):
        for i in range(1, len(word)):
            part1 = word[:i]
            part2 = word[i:]
            if spell_checker.known([part1]) and spell_checker.known([part2]):
                return part1, part2
        return None, None

    def split_words(self, text):
        words = get_words(text)
        if not self.case_sensitive:
            words = [word.lower() for word in words]
        return words

    def get_unknown_words(self, text):
        logger.debug(f"Processing text for unknown words: {text}")
        # Use original text for splitting words to preserve capitalized words
        words = get_words(text)
        logger.debug(f"Split words (original): {words}")
        # Apply remove_proper_nouns to filter true proper nouns
        text_no_proper = self.tagger.remove_proper_nouns(text)
        words_no_proper = get_words(text_no_proper)
        logger.debug(f"Text after removing proper nouns: {text_no_proper}")
        logger.debug(f"Words after removing proper nouns: {words_no_proper}")
        # Use case-insensitive words for spell checking if not case-sensitive
        check_words = [word.lower()
                       for word in words] if not self.case_sensitive else words
        logger.debug(f"Words for spell checking: {check_words}")
        unknown_words_set = self.spell_checker.unknown(check_words)
        logger.debug(
            f"Initial unknown words from spell checker: {unknown_words_set}")
        for word, orig_word in zip(check_words, words):
            count = self.spell_checker._word_frequency.dictionary.get(word, 0)
            logger.debug(
                f"Word '{word}' (original: '{orig_word}') frequency: {count}")
            if count < self.count_threshold:
                # Add original word to preserve case
                unknown_words_set.add(orig_word)
                logger.debug(
                    f"Added '{orig_word}' to unknown words due to low frequency (< {self.count_threshold})")
        unknown_words_list = list(unknown_words_set)
        logger.debug(f"Unknown words before filtering: {unknown_words_list}")
        unknown_words_list = [
            word for word in unknown_words_list if not is_numeric(word)]
        logger.debug(
            f"Unknown words after removing numeric: {unknown_words_list}")
        if self.ignore_words:
            lower_ignore_words = [word.lower() for word in self.ignore_words]
            unknown_words_list = [
                word for word in unknown_words_list if word.lower() not in lower_ignore_words]
            logger.debug(
                f"Unknown words after removing ignore words: {unknown_words_list}")
        logger.debug(f"Final unknown words: {unknown_words_list}")
        # Update the instance variable
        self.unknown_words.update(unknown_words_list)
        logger.debug(f"Updated self.unknown_words: {self.unknown_words}")
        return unknown_words_list

    def autocorrect(self, text: str) -> str:
        """Autocorrect text by replacing misspelled words with the top-ranked suggestion based on final normalized score."""
        words = self.split_words(text)
        misspelled_words = self.get_unknown_words(text)
        if not misspelled_words:
            return text
        suggestions = self.suggest_corrections(misspelled_words, context=text)
        corrected_words_dict = {}
        for word in misspelled_words:
            suggestion_data = suggestions.get(word)
            if suggestion_data and suggestion_data["candidates"]:
                # Select the top candidate based on normalized score
                top_candidate = next(iter(suggestion_data["candidates"]))
                word_for_correction = word.lower() if not self.case_sensitive else word
                if word_for_correction != top_candidate:
                    logger.debug(f"Corrected '{word}' to '{top_candidate}'")
                    corrected_words_dict[word] = top_candidate
        # Replace all occurrences while preserving original case
        corrected_text = text
        for word, corrected_word in sorted(corrected_words_dict.items(), key=lambda x: len(x[0]), reverse=True):
            # Use case-insensitive replacement to handle all variants
            pattern = re.compile(
                r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            corrected_text = pattern.sub(corrected_word, corrected_text)
        return corrected_text

    def suggest_corrections(self, misspelled_words: List[str], context: Optional[str] = None, weights: Optional[WeightsConfig] = None) -> Dict[str, Optional[Dict[str, Dict[str, float]]]]:
        """Suggest corrections for misspelled words, ranked by normalized context similarity, word similarity, and frequency."""
        suggestions = {}
        current_weights = weights or self.weights

        # Validate weights
        required_keys = {"context", "word_similarity", "frequency"}
        if not isinstance(current_weights, dict):
            raise ValueError("Weights must be a dictionary")
        if set(current_weights.keys()) != required_keys:
            raise ValueError(
                f"Weights dictionary must contain exactly these keys: {required_keys}")
        for key, value in current_weights.items():
            if not isinstance(value, (int, float)) or value < 0.0:
                raise ValueError(
                    f"Weight for '{key}' must be a non-negative number, got {value}")
        weight_sum = sum(current_weights.values())
        if not math.isclose(weight_sum, 1.0, rel_tol=1e-5):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

        context_embedding = None
        if context:
            context_embedding = self.sentence_model.encode(context)
            logger.debug(f"Context embedding shape: {context_embedding.shape}")
        for word in misspelled_words:
            word_for_correction = word.lower() if not self.case_sensitive else word
            candidates = self.spell_checker.candidates(word_for_correction)
            candidates = list(candidates) if candidates else []
            if not candidates:
                suggestions[word] = None
                continue
            word_for_embedding = word.lower() if not self.case_sensitive else word
            word_embedding = self.sentence_model.encode(word_for_embedding)
            candidate_embeddings = self.sentence_model.encode(candidates)
            logger.debug(
                f"Word '{word}' embedding shape: {word_embedding.shape}")
            logger.debug(f"Candidates: {candidates}")
            obj = {}
            raw_scores = {}
            context_scores = []
            for candidate, candidate_embedding in zip(candidates, candidate_embeddings):
                word_similarity = util.cos_sim(
                    word_embedding, candidate_embedding)[0].item()
                frequency = self.spell_checker.word_frequency.dictionary.get(
                    candidate, 0)
                frequency_score = frequency / \
                    max(1, self.spell_checker.word_frequency.total_words) * \
                    current_weights["frequency"]
                context_score = 0.0
                if context and context_embedding is not None:
                    temp_context = candidate
                    temp_context_embedding = self.sentence_model.encode(
                        temp_context)
                    context_score = util.cos_sim(context_embedding, temp_context_embedding)[
                        0].item() * current_weights["context"]
                    logger.debug(
                        f"Candidate '{candidate}': context_score={context_score:.3f}, word_similarity={word_similarity:.3f}, frequency_score={frequency_score:.3f}")
                raw_score = context_score + \
                    (word_similarity *
                     current_weights["word_similarity"]) + frequency_score
                # Normalize score (weights sum to 1, so max raw score is roughly sum of max component scores)
                # Since weights sum to 1, raw_score is already in a normalized range
                normalized_score = raw_score
                raw_scores[candidate] = {
                    "context": context_score / current_weights["context"] if current_weights["context"] > 0 else 0.0,
                    "word_similarity": word_similarity,
                    "frequency": frequency / max(1, self.spell_checker.word_frequency.total_words),
                    "frequency_count": frequency,
                    "total": raw_score
                }
                obj[candidate] = normalized_score
                context_scores.append(context_score)
            if context_scores and max(context_scores) - min(context_scores) < 0.05 and current_weights["frequency"] > 0.0:
                for candidate in obj:
                    frequency = self.spell_checker.word_frequency.dictionary.get(
                        candidate, 0)
                    boost = (
                        frequency / max(1, self.spell_checker.word_frequency.total_words)) * 0.15
                    obj[candidate] += boost
                    raw_scores[candidate]["total"] += boost
                    raw_scores[candidate]["frequency_boost"] = boost
                logger.debug(
                    f"Context scores too close for '{word}', boosting frequency: {context_scores}")
            # Sort candidates by normalized score
            obj = dict(
                sorted(obj.items(), key=lambda item: item[1], reverse=True))
            # Assign ranks based on sorted normalized scores
            for rank, candidate in enumerate(obj, 1):
                raw_scores[candidate]["rank"] = rank
            # Assign frequency ranks based on frequency_count
            freq_sorted = sorted(
                raw_scores.items(), key=lambda item: item[1]["frequency_count"], reverse=True)
            current_rank = 1
            previous_freq = None
            for i, (candidate, scores) in enumerate(freq_sorted):
                current_freq = scores["frequency_count"]
                if previous_freq is not None and current_freq < previous_freq:
                    current_rank = i + 1
                raw_scores[candidate]["frequency_rank"] = current_rank
                previous_freq = current_freq
            # Assign context ranks based on context score
            context_sorted = sorted(
                raw_scores.items(), key=lambda item: item[1]["context"], reverse=True)
            current_rank = 1
            previous_context = None
            for i, (candidate, scores) in enumerate(context_sorted):
                current_context = scores["context"]
                if previous_context is not None and current_context < previous_context:
                    current_rank = i + 1
                raw_scores[candidate]["context_rank"] = current_rank
                previous_context = current_context
            # Assign word similarity ranks based on word_similarity score
            word_sorted = sorted(
                raw_scores.items(), key=lambda item: item[1]["word_similarity"], reverse=True)
            current_rank = 1
            previous_word_sim = None
            for i, (candidate, scores) in enumerate(word_sorted):
                current_word_sim = scores["word_similarity"]
                if previous_word_sim is not None and current_word_sim < previous_word_sim:
                    current_rank = i + 1
                raw_scores[candidate]["word_rank"] = current_rank
                previous_word_sim = current_word_sim
            # Sort raw_scores by normalized score to match candidates
            raw_scores = dict(
                sorted(raw_scores.items(), key=lambda item: obj.get(item[0], 0), reverse=True))
            suggestions[word] = {"candidates": obj,
                                 "raw_scores": raw_scores} if obj else None
            logger.debug(f"Suggestions for '{word}': {obj}")
        return suggestions

    def autocorrect_texts(self, data):
        """Autocorrect texts and yield results with context-aware suggestions and raw scores."""
        results = []
        pbar = tqdm(data, desc="Autocorrecting texts")
        for text in pbar:
            misspelled_words = self.get_unknown_words(text)
            corrected_text = self.autocorrect(text)
            if misspelled_words:
                suggested_corrections = self.suggest_corrections(
                    misspelled_words, context=text)
                result = {
                    "original": text,
                    "corrected": corrected_text,
                    "suggestions": suggested_corrections
                }
                results.append(result)
                pbar.set_description_str(f"Misspelled: ({len(results)})")
                yield result
        return results
