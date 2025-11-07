from enum import Enum
import threading
import spacy
import json
from collections import defaultdict
from jet.wordnet.sentence import split_sentences
from jet.file.utils import load_data
from typing import Literal, TypedDict, Union


POSTag = Literal[
    "PROPN",  # Proper noun
    "NOUN",   # Noun
    "VERB",   # Verb
    "ADJ",    # Adjective
    "ADV",    # Adverb
    "PRON",   # Pronoun
    "DET",    # Determiner
    "ADP",    # Adposition
    "AUX",    # Auxiliary verb
    "SCONJ",  # Subordinating conjunction
    "CCONJ",  # Coordinating conjunction
    "NUM",    # Numeral
    "PART",   # Particle
    "INTJ",   # Interjection
    "PUNCT",  # Punctuation
    "SYM",    # Symbol
    "X",      # Other
]


class POSTagEnum(Enum):
    PROPN = "PROPN"  # Proper noun
    NOUN = "NOUN"    # Noun
    VERB = "VERB"    # Verb
    ADJ = "ADJ"      # Adjective
    ADV = "ADV"      # Adverb
    PRON = "PRON"    # Pronoun
    DET = "DET"      # Determiner
    ADP = "ADP"      # Adposition
    AUX = "AUX"      # Auxiliary verb
    SCONJ = "SCONJ"  # Subordinating conjunction
    CCONJ = "CCONJ"  # Coordinating conjunction
    NUM = "NUM"      # Numeral
    PART = "PART"    # Particle
    INTJ = "INTJ"    # Interjection
    PUNCT = "PUNCT"  # Punctuation
    SYM = "SYM"      # Symbol
    X = "X"          # Other


POSTagType = Union[POSTag, POSTagEnum, str]


class POSItem(TypedDict):
    word: str
    pos: list[POSTagType]  # Using Enum for controlled POS tags


class POSTagger:
    _instance = None
    _instance_lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        # Ensure singleton instance creation is thread-safe
        with cls._instance_lock:
            if not cls._instance:
                cls._instance = super(POSTagger, cls).__new__(cls)
        return cls._instance

    def __init__(self, dictionary_file=None, spacy_model_name="en_core_web_md"):
        # Only initialize once, even under concurrent access
        if not hasattr(self, '_initialized'):
            with self._instance_lock:
                if not hasattr(self, '_initialized'):
                    self.dictionary_file = dictionary_file
                    self.spacy_model_name = spacy_model_name
                    self.nlp_models = {}
                    self.cache = load_data(dictionary_file) if dictionary_file else defaultdict(dict)
                    self._lock = threading.RLock()  # Protects cache & model
                    self._initialized = True

    def load_model(self):
        with self._lock:
            if 'en' not in self.nlp_models:
                self.nlp_models['en'] = spacy.load(self.spacy_model_name)
            return self.nlp_models['en']

    def process_and_tag(self, text) -> list[POSItem]:
        # Split text into sentences
        sentences = split_sentences(text)

        combined_pos_results = []
        for sentence in sentences:
            pos_results = self.tag_string(sentence)
            updated_pos_results = self.merge_multi_word_pos(pos_results)
            combined_pos_results.extend(updated_pos_results)

        return combined_pos_results

    def tag_word(self, word):
        pos_results = self.tag_string(word)
        return pos_results[0]['pos']

    def tag_string(self, string) -> list[POSItem]:
        # Safe cache read/write
        with self._lock:
            if string in self.cache['en']:
                return self.cache['en'][string]

        nlp = self.load_model()
        doc = nlp(string)
        pos_results = [{'word': token.text, 'pos': token.pos_} for token in doc]

        # Write to cache under lock
        with self._lock:
            self.cache['en'][string] = pos_results

        return pos_results

    def merge_multi_word_pos(self, pos_results: list[POSItem]) -> list[POSItem]:
        merged_results: list[POSItem] = []
        i = 0
        while i < len(pos_results):
            current_word = pos_results[i]['word']
            current_word_pos = pos_results[i]['pos']
            is_current_word_noun = current_word_pos in ['PROPN', 'NOUN']

            if current_word in ['The', 'the']:
                current_word_pos = 'DET'
                merged_results.append(
                    {'word': current_word, 'pos': current_word_pos})
                i += 1

            elif (pos_results[i + 1]['word'] if i + 1 < len(pos_results) else None) == '-':
                current_word = pos_results[i]['word']
                current_word_pos = pos_results[i]['pos']

                i += 1

                while i < len(pos_results):
                    next_word = pos_results[i]['word']
                    if next_word == '-':
                        if i + 1 < len(pos_results):
                            hyphen_next_word = pos_results[i + 1]['word']
                            hyphen_next_word_pos = pos_results[i + 1]['pos']

                            current_word += next_word + hyphen_next_word

                            if current_word_pos == hyphen_next_word_pos:
                                current_word_pos = hyphen_next_word_pos
                            elif not is_current_word_noun:
                                current_word_pos = 'VERB'
                            elif current_word_pos == 'NOUN' and hyphen_next_word_pos == 'PROPN':
                                current_word_pos = 'PROPN'
                            elif current_word_pos == 'PROPN' and hyphen_next_word_pos == 'NOUN':
                                current_word_pos = 'NOUN'
                            else:
                                hyphen_next_word_pos = self.tag_word(
                                    hyphen_next_word)

                            i += 2  # Skip the next word as it has been processed
                        else:
                            break
                    else:
                        break

                merged_results.append(
                    {'word': current_word, 'pos': current_word_pos})

            else:
                merged_results.append(pos_results[i])
                i += 1

        return merged_results

    def tag_text(self, text):
        pos_results = self.process_and_tag(text)
        tagged_text = self.format_tags(pos_results)
        return tagged_text

    def format_tags(self, pos_results):
        tagged_texts = []
        for pair in pos_results:
            tagged_texts.append(f"{pair['word']}/{pair['pos']}")

        return " ".join(tagged_texts)

    def remove_proper_nouns(self, text) -> str:
        pos_results = self.process_and_tag(text)
        proper_nouns = [pos_result['word']
                        for pos_result in pos_results if pos_result['pos'] == 'PROPN']

        # Remove proper nouns from text using replace
        for proper_noun in proper_nouns:
            text = text.replace(proper_noun, '')

        return text.strip()

    def contains_pos(self, text: str, pos: str | list[str]) -> bool:
        pos_results = self.process_and_tag(text)
        pos_lower = [p.lower() for p in pos]
        if isinstance(pos, str):
            pos_lower = [pos.lower()]

        has_pos = any([pos_result['pos'].lower()
                      in pos_lower for pos_result in pos_results])

        return has_pos

    def validate_pos(self, text: str, pos_index_mapping: dict) -> bool:
        pos_results = self.process_and_tag(text)
        passed = True

        for word_index, pos_dict in pos_index_mapping.items():
            if not passed:
                break

            word_index = int(word_index)
            excludes_pos = pos_dict.get('excludes', [])
            includes_pos = pos_dict.get('includes', [])

            if word_index >= len(pos_results):
                raise ValueError(
                    f"Word index {word_index} is out of range for text '{text}'")

            pos = pos_results[word_index]['pos']
            word = pos_results[word_index]['word']
            if includes_pos and (pos not in includes_pos and word not in includes_pos):
                passed = False
                break
            if excludes_pos and (pos in excludes_pos or word in excludes_pos):
                passed = False
                break

        return passed

    def validate_equal_pos(self, tl_text: str, en_text: str, pos: str | list[str]) -> bool:
        en_pos_results = self.filter_pos(en_text, includes=pos)

        if isinstance(pos, str):
            pos = [pos]

        filtered_en_pos = [pos_result['pos'] for pos_result in en_pos_results]

        return filtered_en_pos == filtered_en_pos

    def filter_pos(self, text: str, includes: str | list[str] = None, excludes: str | list[str] = None) -> list[POSItem]:
        pos_results = self.process_and_tag(text)
        filtered_pos = [pos_result for pos_result in pos_results if
                        (not includes or pos_result['pos'] in includes) and
                        (not excludes or pos_result['pos'] not in excludes)
                        ]

        return filtered_pos

    def filter_words(self, text: str, includes: str | list[str] = None, excludes: str | list[str] = None) -> str:
        pos_results = self.process_and_tag(text)
        filtered_words = [pos_result['word']
                          for pos_result in pos_results if
                          (not includes or pos_result['pos'] in includes) and
                          (not excludes or pos_result['word'] not in excludes)
                          ]

        return " ".join(filtered_words)


if __name__ == '__main__':
    tagger = POSTagger()
    all_texts = []

    texts = [
        "Dr. Jose Rizal is the only example of a genius in many fields who became the greatest hero of a nation",
        # "Which then spawned the short-lived First Philippine Republic.",
        # "It's more fun in Republic of the Congo."
    ]

    print("Tagging Words:")
    for text in texts:
        pos_results = tagger.process_and_tag(text)
        tagged_text = tagger.format_tags(pos_results)
        merged_results = tagger.merge_multi_word_pos(pos_results)

        print(f"Tagged Text:\n{tagged_text}")
        print(
            f"POS Results:\n{json.dumps(pos_results, indent=2, ensure_ascii=False)}")
        print(
            f"Merged Results:\n{json.dumps(merged_results, indent=2, ensure_ascii=False)}")
