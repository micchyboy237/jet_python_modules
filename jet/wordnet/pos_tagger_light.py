import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from typing import Literal, Optional, Set, TypedDict, Union, List
from collections import defaultdict
from jet.file.utils import load_data
from jet.wordnet.sentence import split_sentences
from enum import Enum
from jet.scrapers.utils import clean_newlines, clean_punctuations, clean_spaces
from jet.search.formatters import clean_string
from jet.wordnet.words import get_words

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Existing POSTag and POSTagEnum definitions remain unchanged
POSTag = Literal[
    "PROPN", "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "AUX",
    "SCONJ", "CCONJ", "NUM", "PART", "INTJ", "PUNCT", "SYM", "X"
]


class POSTagEnum(Enum):
    PROPN = "PROPN"
    NOUN = "NOUN"
    VERB = "VERB"
    ADJ = "ADJ"
    ADV = "ADV"
    PRON = "PRON"
    DET = "DET"
    ADP = "ADP"
    AUX = "AUX"
    SCONJ = "SCONJ"
    CCONJ = "CCONJ"
    NUM = "NUM"
    PART = "PART"
    INTJ = "INTJ"
    PUNCT = "PUNCT"
    SYM = "SYM"
    X = "X"


POSTagType = Union[POSTag, POSTagEnum, str]


class POSItem(TypedDict):
    word: str
    pos: POSTagType


class POSTagger:
    _instance = None

    # Mapping NLTK Penn Treebank tags to SpaCy-compatible tags
    NLTK_TO_SPACY = {
        'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'PROPN', 'NNPS': 'PROPN',
        'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 'VBP': 'VERB', 'VBZ': 'VERB',
        'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
        'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV',
        'PRP': 'PRON', 'PRP$': 'PRON',
        'DT': 'DET',
        'IN': 'ADP',
        'MD': 'AUX',
        'CC': 'CCONJ',
        'CD': 'NUM',
        'TO': 'PART',
        'UH': 'INTJ',
        '.': 'PUNCT', ',': 'PUNCT', ':': 'PUNCT', '``': 'PUNCT', "''": 'PUNCT',
        '$': 'SYM', '#': 'SYM',
        'FW': 'X', 'POS': 'X',
        'EX': 'PRON',  # Existential "there" maps to pronoun
        'LS': 'X',     # List item marker, no direct SpaCy equivalent
        'PDT': 'DET',  # Predeterminer maps to determiner
        'RP': 'PART',  # Particle maps to particle
        'WDT': 'DET',  # Wh-determiner maps to determiner
        'WP': 'PRON',  # Wh-pronoun maps to pronoun
        'WP$': 'PRON',  # Possessive wh-pronoun maps to pronoun
        'WRB': 'ADV',  # Wh-adverb maps to adverb
        '-LRB-': 'PUNCT', '-RRB-': 'PUNCT',  # Parentheses map to punctuation
        '"': 'PUNCT'   # Double quote maps to punctuation
    }

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(POSTagger, cls).__new__(cls)
        return cls._instance

    def __init__(self, dictionary_file=None):
        if not hasattr(self, '_initialized'):
            self.dictionary_file = dictionary_file
            self.cache = load_data(
                dictionary_file) if dictionary_file else defaultdict(dict)
            self._initialized = True

    def tag_string(self, string: str) -> List[POSItem]:
        if string in self.cache.get('en', {}):
            return self.cache['en'][string]

        tokens = word_tokenize(string)
        nltk_tags = nltk.pos_tag(tokens)
        pos_results = [
            {'word': word, 'pos': self.NLTK_TO_SPACY.get(tag, 'X')}
            for word, tag in nltk_tags
        ]

        self.cache['en'][string] = pos_results
        return pos_results

    def process_and_tag(self, text: str) -> List[POSItem]:
        sentences = split_sentences(text)
        combined_pos_results = []
        for sentence in sentences:
            pos_results = self.tag_string(sentence)
            updated_pos_results = self.merge_multi_word_pos(pos_results)
            combined_pos_results.extend(updated_pos_results)
        return combined_pos_results

    def tag_word(self, word: str) -> POSTagType:
        pos_results = self.tag_string(word)
        return pos_results[0]['pos'] if pos_results else 'X'

    def merge_multi_word_pos(self, pos_results: List[POSItem]) -> List[POSItem]:
        merged_results: List[POSItem] = []
        i = 0
        while i < len(pos_results):
            current_word = pos_results[i]['word']
            current_word_pos = pos_results[i]['pos']
            is_current_word_noun = current_word_pos in ['PROPN', 'NOUN']

            if current_word.lower() in ['the']:
                current_word_pos = 'DET'
                merged_results.append(
                    {'word': current_word, 'pos': current_word_pos})
                i += 1
            elif (i + 1 < len(pos_results) and pos_results[i + 1]['word'] == '-'):
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
                                current_word_pos = self.tag_word(
                                    hyphen_next_word)

                            i += 2
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

    def format_tags(self, pos_results: List[POSItem]) -> str:
        tagged_texts = [
            f"{pair['word']}/{pair['pos']}" for pair in pos_results]
        return " ".join(tagged_texts)

    def remove_proper_nouns(self, text: str) -> str:
        pos_results = self.process_and_tag(text)
        proper_nouns = [pos_result['word']
                        for pos_result in pos_results if pos_result['pos'] == 'PROPN']
        for proper_noun in proper_nouns:
            text = text.replace(proper_noun, '')
        return text.strip()

    def contains_pos(self, text: str, pos: Union[str, List[str]]) -> bool:
        pos_results = self.process_and_tag(text)
        pos_lower = [p.lower()
                     for p in ([pos] if isinstance(pos, str) else pos)]
        return any(pos_result['pos'].lower() in pos_lower for pos_result in pos_results)

    def validate_pos(self, text: str, pos_index_mapping: dict) -> bool:
        pos_results = self.process_and_tag(text)
        for word_index, pos_dict in pos_index_mapping.items():
            word_index = int(word_index)
            excludes_pos = pos_dict.get('excludes', [])
            includes_pos = pos_dict.get('includes', [])
            if word_index >= len(pos_results):
                raise ValueError(
                    f"Word index {word_index} is out of range for text '{text}'")
            pos = pos_results[word_index]['pos']
            word = pos_results[word_index]['word']
            if includes_pos and (pos not in includes_pos and word not in includes_pos):
                return False
            if excludes_pos and (pos in excludes_pos or word in excludes_pos):
                return False
        return True

    def filter_pos(self, text: str, includes: Union[str, List[str]] = None, excludes: Union[str, List[str]] = None) -> List[POSItem]:
        pos_results = self.process_and_tag(text)
        includes = [includes] if isinstance(includes, str) else includes
        excludes = [excludes] if isinstance(excludes, str) else excludes
        return [
            pos_result for pos_result in pos_results
            if (not includes or pos_result['pos'] in includes) and
            (not excludes or pos_result['pos'] not in excludes)
        ]

    def filter_words(self, text: str, includes: Union[str, List[str]] = None, excludes: Union[str, List[str]] = None) -> str:
        pos_results = self.process_and_tag(text)
        includes = [includes] if isinstance(includes, str) else includes
        excludes = [excludes] if isinstance(excludes, str) else excludes
        filtered_words = [
            pos_result['word'] for pos_result in pos_results
            if (not includes or pos_result['pos'] in includes) and
               (not excludes or pos_result['word'] not in excludes)
        ]
        return " ".join(filtered_words)


def preprocess_texts(texts: Union[str, List[str]]) -> List[str]:
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    if isinstance(texts, str):
        texts = [texts]

    texts = [text.lower() for text in texts]
    preprocessed_texts: List[str] = texts.copy()
    stop_words = set(stopwords.words('english'))
    tagger = POSTagger()

    for idx, text in enumerate(tqdm(preprocessed_texts, desc="Preprocessing texts")):
        includes_pos = ["PROPN", "NOUN", "VERB", "ADJ", "ADV"]
        text = clean_newlines(text, max_newlines=1)
        text = clean_punctuations(text)
        text = clean_spaces(text)
        text = clean_string(text)

        preprocessed_lines = []
        for line in text.splitlines():
            pos_results = tagger.filter_pos(line, includes=includes_pos)
            filtered_text = [pos_result['word'] for pos_result in pos_results]
            text = " ".join(filtered_text).lower()

            words = get_words(text)
            filtered_words = [
                word for word in words if word.lower() not in stop_words]
            preprocessed_lines.append(' '.join(filtered_words))
        preprocessed_texts[idx] = '\n'.join(preprocessed_lines)

    return preprocessed_texts


if __name__ == '__main__':
    import json

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
