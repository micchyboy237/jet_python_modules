import json
import os
import re
from spellchecker import SpellChecker
from tqdm import tqdm
from jet.file.utils import load_data, save_data
from jet.logger import logger
from jet.utils.string_utils import is_numeric
from jet.wordnet.pos_tagger_light import POSTagger
from jet.wordnet.words import get_words


class SpellingCorrector:
    def __init__(self, *, dictionary_file=None, case_sensitive=False, ignore_words=None, base_words=None, base_sentences=None,  count_threshold=5):
        self.spell_checker = SpellChecker()
        self.tagger = POSTagger()

        self.case_sensitive = case_sensitive
        self.base_words = base_words or list(
            self.spell_checker.word_frequency.words())
        self.ignore_words = ignore_words + \
            self.base_words if ignore_words else self.base_words
        self.count_threshold = count_threshold

        self.unknown_words = set()

        data = base_sentences

        logger.info(f"Base words: {len(self.base_words)}")
        logger.info(f"Ignore words: {len(self.ignore_words)}")

        if not data and dictionary_file and os.path.exists(dictionary_file):
            logger.info(f"Loading dictionary from {dictionary_file}")
            self.spell_checker.word_frequency.load_json(
                load_data(dictionary_file, is_binary=True))
        elif data:
            # logger.info(
            #     "Getting data from tagged data without proper nouns")
            # result = get_word_pos_dict(excludes_pos=["PROPN"])
            # data = result['texts']

            logger.info("Loading data and updating spellchecker")
            self.load_data_and_update_spellchecker(data)

            if dictionary_file:
                logger.info(f"Saving dictionary to {dictionary_file}")
                self.save_dictionary(dictionary_file)

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
        save_data(dictionary_file,
                  self.spell_checker.word_frequency.dictionary, overwrite=True, is_binary=True)

    def split_compound_word(self, word):
        for i in range(1, len(word)):
            part1 = word[:i]
            part2 = word[i:]
            if self.spell_checker.known([part1]) and self.spell_checker.known([part2]):
                return part1, part2
        return None, None

    def split_words(self, text):
        words = get_words(text)

        # Adjust for case sensitivity
        if not self.case_sensitive:
            words = [word.lower() for word in words]

        return words

    def get_unknown_words(self, text):
        # text = self.tagger.remove_proper_nouns(text)
        words = self.split_words(text)
        unknown_words_set = self.spell_checker.unknown(words)

        for word in words:
            count = self.spell_checker._word_frequency.dictionary[word]

            if count < self.count_threshold:
                # Add to unknown words
                unknown_words_set.add(word)

        unknown_words_list = list(unknown_words_set)

        # Remove numeric words
        unknown_words_list = [
            word for word in unknown_words_list if not is_numeric(word)]

        # Remove words in ignore words
        if self.ignore_words:
            # lowercase the ignore words and the unknown words
            lower_ignore_words = [word.lower() for word in self.ignore_words]
            # remove words from unknown words that are in ignore words
            unknown_words_list = [
                word for word in unknown_words_list if word.lower() not in lower_ignore_words]

        self.unknown_words.update(unknown_words_list)
        return unknown_words_list

    def get_word_indices(self, text, word):
        """Find all occurrences of a word in the text with their start and end indices."""
        indices = []
        words = get_words(text)
        current_pos = 0

        for w in words:
            # Find the next occurrence of the word starting from current_pos
            start = text.find(w, current_pos)
            if start == -1:
                continue  # Word not found, move to next
            end = start + len(w)

            # Check if the word matches, respecting case sensitivity
            if self.case_sensitive:
                if w == word:
                    indices.append({"start": start, "end": end})
            else:
                if w.lower() == word.lower():
                    indices.append({"start": start, "end": end})

            current_pos = start + 1  # Move past the current character to find next occurrence

        return indices

    def autocorrect(self, text: str) -> tuple[str, list[dict]]:
        words = self.split_words(text)
        corrected_words_dict = {}
        corrections_with_indices = []

        # Track unique words to handle repeats
        processed_words = set()

        for word in words:
            if word in processed_words:
                continue  # Skip already processed words to avoid duplicate corrections
            corrected_word = self.spell_checker.correction(word)

            if corrected_word is None:
                continue  # Skip if no correction is available

            if word != corrected_word:
                corrected_words_dict[word] = corrected_word
                indices = self.get_word_indices(text, word)
                for idx in indices:
                    corrections_with_indices.append({
                        "original": word,
                        "corrected": corrected_word,
                        "start": idx["start"],
                        "end": idx["end"]
                    })
                processed_words.add(word)

        # Replace text with corrected words, preserving original case where possible
        corrected_text = text
        for word, corrected_word in corrected_words_dict.items():
            if self.case_sensitive:
                corrected_text = corrected_text.replace(word, corrected_word)
            else:
                # Replace case-insensitively while preserving the original case
                pattern = re.compile(
                    r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                corrected_text = pattern.sub(corrected_word, corrected_text)

        return corrected_text, corrections_with_indices

    def suggest_corrections(self, misspelled_words):
        """Suggest corrections for the identified misspelled words."""
        suggestions = {}
        for word in misspelled_words:
            candidates = self.spell_checker.candidates(word)
            candidates = list(candidates) if candidates else None
            # obj is dict with keys of candidates and values of their frequency using word_usage_frequency
            obj = None
            if candidates:
                for candidate in candidates:
                    if not obj:
                        obj = {}
                    dictionary = self.spell_checker.word_frequency.dictionary

                    if dictionary[candidate]:
                        obj[candidate] = dictionary[candidate]

                # Sort the dictionary by frequency in descending order
                if obj:
                    obj = dict(
                        sorted(obj.items(), key=lambda item: item[1], reverse=True))
                else:
                    obj = None

            suggestions[word] = obj
        return suggestions

    def autocorrect_texts(self, data):
        results = []

        pbar = tqdm(data, desc="Autocorrecting texts")
        for text in pbar:
            misspelled_words = self.get_unknown_words(text)

            if misspelled_words:
                suggested_corrections = self.suggest_corrections(
                    misspelled_words)

                corrected_text = self.autocorrect(text)

                result = {
                    "original": text,
                    "corrected_text": corrected_text,
                    "suggestions": suggested_corrections,
                }

                results.append(result)

                pbar.set_description_str(f"Misspelled: ({len(results)})")

                yield result
        return results


if __name__ == '__main__':
    text_corrector = SpellingCorrector()
    english_texts = [
        "Hello, world! I am fine.",
        "She is he or she."
    ]
    data_texts = english_texts

    results_stream = text_corrector.autocorrect_texts(data_texts)

    for result in results_stream:
        print(f"Result: {json.dumps(result, indent=2)}")
