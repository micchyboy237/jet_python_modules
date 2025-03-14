from collections import Counter
from instruction_generator.utils.time_utils import time_it
from instruction_generator.wordnet.constants import PREFIX_SET
from instruction_generator.helpers.words import get_words
from instruction_generator.wordnet.pos_tagger import POSTagger
from instruction_generator.helpers.dataset import save_data
from instruction_generator.analyzers.helpers import (
    get_language_sentences, get_english_words, get_tagalog_words)
from tqdm import tqdm


class TaglishAnalyzer:
    @time_it
    def __init__(self):
        self.prefixes = PREFIX_SET
        self.tagalog_words = get_tagalog_words(
            count_threshold=5, excludes_pos=["PART"])
        self.english_words = get_english_words(
            count_threshold=10,
            includes_pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'NUM', 'PRON'])
        self.tagger = POSTagger()

    @time_it
    def analyze_taglish_patterns(self, sentences):
        english_counts = Counter()
        tagalog_counts = Counter()
        for sentence in sentences:
            words = get_words(sentence.lower())
            for word in words:
                if word in self.english_words:
                    english_counts[word] += 1
                elif word in self.tagalog_words:
                    tagalog_counts[word] += 1

        return english_counts, tagalog_counts

    @time_it
    def evaluate_taglish_sentences(self, sentences):
        sentences = sentences.copy()

        potential_synthesis = []
        potential_count = 0

        pbar = tqdm(sentences, desc="Evaluating sentences")
        for sentence in pbar:
            words = get_words(sentence.lower())
            english_only_words = [
                word for word in words if word in self.english_words and word not in self.tagalog_words]
            taglish_potential = len(english_only_words)

            affixed_words = self.get_affixed_words([sentence])
            if affixed_words:
                taglish_potential += len(affixed_words)

            if taglish_potential > 0:
                words_list = []
                if english_only_words:
                    words_list.extend([{
                        "word": word,
                        "type": "english"
                    } for word in english_only_words])
                if affixed_words:
                    words_list.extend([{
                        "word": word,
                        "prefix": prefix,
                        "type": "affixed"
                    } for prefix, word in affixed_words])

                result = {
                    "sentence": sentence,
                    "words": words_list,
                    "taglish_potential": taglish_potential
                }
                potential_synthesis.append(result)
                potential_count += 1

                pbar.set_postfix_str(f"Potential: {potential_count}")

                yield result

            pbar.update(1)
        return potential_synthesis

    def get_affixed_words(self, sentences):
        affixed_words = {}

        # Initialize dictionary to hold lists for each prefix
        for prefix in self.prefixes:
            affixed_words[prefix] = []

        for sentence in sentences:
            words = get_words(sentence.lower())
            for word in words:
                for prefix in self.prefixes:
                    if word.startswith(prefix) and word not in self.tagalog_words:
                        # Check if the word without prefix is in the list of English words
                        if word[len(prefix):] in self.english_words or prefix + word in self.english_words:
                            affixed_words[prefix].append(word)
                            break  # Stop checking other prefixes once a match is found

        # Filter out prefixes with no affixed words found
        affixed_words = {prefix: words for prefix,
                         words in affixed_words.items() if words}
        return affixed_words

    @time_it
    def get_taglish_sentences(self, sentences):
        taglish_sentences = []

        for sentence in sentences:
            words = get_words(sentence.lower())
            for word in words:
                for prefix in self.prefixes:
                    if word.startswith(prefix) and word[len(prefix):] in self.english_words:
                        taglish_sentences.append(sentence)
                        break  # Stop processing this sentence once a match is found

        return taglish_sentences


if __name__ == "__main__":
    taglish_sentences_file = 'instruction_generator/wordnet/datasets/taglish_sentences_scores.json'

    analyzer = TaglishAnalyzer()

    # Analyze Taglish patterns
    data = get_language_sentences()
    tl_sentences = data["tl"]

    potential_synthesis_stream = analyzer.evaluate_taglish_sentences(
        tl_sentences)

    batch_size = 100
    batch_items = []
    for result in potential_synthesis_stream:
        batch_items.append(result)
        if len(batch_items) >= batch_size:
            save_data(taglish_sentences_file, batch_items, key="sentence")
            batch_items = []

    if batch_items:
        save_data(taglish_sentences_file, batch_items, key="sentence")

    english_counts, tagalog_counts = analyzer.analyze_taglish_patterns(
        tl_sentences)
    print(f"English word counts: {english_counts}")
    print(f"Tagalog word counts: {tagalog_counts}")
