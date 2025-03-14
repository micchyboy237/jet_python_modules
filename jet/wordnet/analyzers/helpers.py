from collections import OrderedDict
import re
import os
from jet.file.utils import load_data, load_data_from_directories
from typing import Optional, List
from jet.search.transformers import decode_encoded_characters
from tqdm import tqdm
from jet.logger import time_it
from jet.wordnet.words import get_words
from gensim.corpora import Dictionary

UNIGRAMS_FILE = "server/static/unigrams/unigrams.bin"
DICTIONARY_FILE = "server/static/unigrams/en_tl_dictionary.bin"
TAGGED_DATA_FILE = "server/static/unigrams/tagged_data.bin"
ENGLISH_UNIGRAMS_POS_WORD_FILE = "instruction_generator/analyzers/datasets/en/1_word_dict_all.json"
TAGALOG_UNIGRAMS_POS_WORD_FILE = "instruction_generator/analyzers/datasets/tl/1_word_dict_all.json"
TAGALOG_START_POS_WORD_FILE = "instruction_generator/analyzers/datasets/tl/1_word_dict_start.json"
ENGLISH_START_POS_WORD_FILE = "instruction_generator/analyzers/datasets/en/1_word_dict_start.json"
TAGALOG_AFFIXED_WORDS_FILE = "instruction_generator/wordnet/datasets/affix_data_tl_translations.json"
TRANSLATION_FILES = [
    'server/static/models/dost-asti-gpt2/base_model/datasets/base/tagalog_phrases_translations.json',
    'server/static/datasets/translations/learnentry.com.json',
    'server/static/datasets/translations/jmbt22_tatoeba-en-tgl.json',
    'server/static/datasets/translations/greetings_translation.json',
    'server/static/models/dost-asti-gpt2/base_model/datasets/foundational1/youtube_translations.json',
    'server/static/models/dost-asti-gpt2/base_model/datasets/foundational1/youtube_translation_pairs.json',
    'server/static/datasets/translations/filwords_wiki_corpus_en_tl.json',
    # 'server/static/datasets/translations/diwa_translation.json',
    # 'server/static/datasets/translations/alpaca_gpt4_instructions_input_en_tl.json',
    # 'server/static/datasets/translations/alpaca_gpt4_outputs_en_tl.json',
    # 'server/static/models/dost-asti-gpt2/base_model/datasets/base/glosbe_top_translations.json',
    # 'server/static/datasets/translations/Ramos-Ramos_nllb-eng-tgl-600k.json',
]


def get_base_words():
    file_path = UNIGRAMS_FILE
    data = load_data(file_path, is_binary=True)
    return data


def get_language_sentences():
    files = TRANSLATION_FILES
    data = load_data_from_directories(files)
    tl_data = [item['translation.tl'] for item in data]
    en_data = [item['translation.en'] for item in data]

    return {
        "tl": tl_data,
        "en": en_data
    }


def get_translation_sentences():
    files = TRANSLATION_FILES
    data = load_data_from_directories(files)
    sentences = []

    for item in data:
        # sentences.append(item['translation.tl'].lower())
        # sentences.append(item['translation.en'].lower())
        sentences.append(item['translation.tl'])
        sentences.append(item['translation.en'])

    return sentences


def get_tl_translation_sentences():
    files = TRANSLATION_FILES
    data = load_data_from_directories(files)
    sentences = []

    for item in data:
        sentences.append(item['translation.tl'])

    return sentences


def get_en_translation_sentences():
    files = TRANSLATION_FILES
    data = load_data_from_directories(files)
    sentences = []

    for item in data:
        sentences.append(item['translation.en'])

    return sentences


@time_it
def get_tl_corpus_sentences():
    files_dir = [
        "server/static/models/dost-asti-gpt2/base_model/datasets/base/tagalog_corpus_results/balita_nlp",
        "server/static/models/dost-asti-gpt2/base_model/datasets/base/tagalog_corpus_results/tl_unified",
        "server/static/models/dost-asti-gpt2/base_model/datasets/base/tagalog_corpus_results/wikitext_tl",
    ]
    includes = ["train_results.json",
                "test_results.json", "validation_results.json"]
    data = load_data_from_directories(files_dir, includes=includes)
    sentences = []
    for item in data:
        sentences.extend(item['texts'])

    taglish_file = 'server/static/models/dost-asti-gpt2/base_model/datasets/base/reddit_posts.json'
    data = load_data(taglish_file)
    texts = [item['text'] for sublist in data for item in sublist['posts']]
    texts = list(set(texts))
    sentences.extend(texts)

    return sentences


def create_dictionary_counts(texts):
    texts = [get_words(normalize_text(text.lower()))
             for text in texts]
    dictionary = Dictionary(texts)
    dictionary_counts = {
        dictionary[word_id]: doc_freq  # / len(texts)
        for word_id, doc_freq in dictionary.dfs.items()
        # if doc_freq / len(texts) >= frequency_threshold
    }
    # Sort the words by frequency in descending order
    dictionary_counts = {
        word: freq for word, freq in sorted(
            dictionary_counts.items(), key=lambda item: item[1], reverse=True)
    }
    return dictionary_counts


def normalize_text(text: str) -> str:
    """Decode and remove hyphens from a text to normalize it."""
    text = decode_encoded_characters(text)
    text = re.sub(r"-", "", text)
    return text


def get_tl_lemmas_counts_dict() -> dict[str, int]:
    affixed_words_file = 'instruction_generator/wordnet/datasets/affix_data_tl_translations.json'
    all_lemmas = load_data(affixed_words_file)
    all_lemmas_word_count = {}
    all_items = (list(all_lemmas.get("prefix", {}).items()) +
                 list(all_lemmas.get("infix", {}).items()) +
                 list(all_lemmas.get("suffix", {}).items()))

    for _, item in all_items:
        word_counts = item['words'].items()
        for word, count in word_counts:
            normalized_word = word.lower()
            if normalized_word in all_lemmas_word_count:
                all_lemmas_word_count[normalized_word] += count
            else:
                all_lemmas_word_count[normalized_word] = count

    return all_lemmas_word_count


def sort_dictionary(dictionary: dict) -> dict:
    # Sorting by word length and then alphabetically
    return OrderedDict(
        sorted(dictionary.items(), key=lambda x: (len(x[0]), x[0]))
    )


def get_tl_first_words(count_threshold=2, includes_pos=None, excludes_pos=None) -> dict[str, int]:
    file_path = TAGALOG_START_POS_WORD_FILE
    data_dict = load_data(file_path)
    filtered_data_dict = {}
    for pos, pos_items in data_dict.items():
        for word, count in pos_items.items():
            if (not includes_pos or pos in includes_pos) and (not excludes_pos or pos not in excludes_pos):
                if count >= count_threshold and (word.isalpha() or word.isalnum()):
                    filtered_data_dict[word] = count
    # Sort by count in descending order
    sorted_data_dict = dict(sorted(filtered_data_dict.items(),
                                   key=lambda item: item[1], reverse=True))
    return sorted_data_dict


def get_en_first_words(count_threshold=10, includes_pos=None, excludes_pos=None) -> dict[str, int]:
    file_path = ENGLISH_START_POS_WORD_FILE
    data_dict = load_data(file_path)
    filtered_data_dict = {}
    for pos, pos_items in data_dict.items():
        for word, count in pos_items.items():
            if (not includes_pos or pos in includes_pos) and (not excludes_pos or pos not in excludes_pos):
                if count >= count_threshold and (word.isalpha() or word.isalnum()):
                    filtered_data_dict[word] = count
    # Sort by count in descending order
    sorted_data_dict = dict(sorted(filtered_data_dict.items(),
                                   key=lambda item: item[1], reverse=True))
    return sorted_data_dict


@time_it
def get_tl_dictionary(min_count=2, includes_pos=None, excludes_pos=None) -> dict[str, int]:
    # texts = get_tl_translation_sentences()
    # dictionary = create_dictionary_counts(texts)
    dictionary = {}

    # Add lemmas to dictionary if not already in dictionary
    # lemmas_counts_dict = get_tl_lemmas_counts_dict()
    # for lemma, count in lemmas_counts_dict.items():
    #     if lemma not in dictionary:
    #         dictionary[lemma] = count

    # Add words to dictionary if not already in dictionary
    tagalog_words_info_dict = get_tagalog_word_info_dict(
        count_threshold=min_count, includes_pos=includes_pos, excludes_pos=excludes_pos)
    for word, info in tagalog_words_info_dict.items():
        word_count = dictionary.get(word, 0)
        if info['count'] > word_count:
            dictionary[word] = info['count']

    # Create a list of words to delete based on the conditions
    words_to_delete = [
        word for word, count in dictionary.items() if len(word) < 3 or (min_count and count < min_count) or not word.isalpha()]

    # Delete the words from the dictionary
    for word in words_to_delete:
        del dictionary[word]

    # Sorting by word length and then alphabetically
    sorted_dictionary = OrderedDict(
        sorted(dictionary.items(), key=lambda x: (len(x[0]), x[0])))
    return sorted_dictionary


@time_it
def get_en_dictionary():
    texts = get_en_translation_sentences()
    dictionary = create_dictionary_counts(texts)
    return dictionary


def get_word_dictionary():
    file_path = DICTIONARY_FILE
    data = load_data(file_path, is_binary=True)
    return data


def get_tagged_data():
    file_path = TAGGED_DATA_FILE
    data = load_data(file_path, is_binary=True)
    return data


def get_tagged_data_dict(includes_pos=None, excludes_pos=None):
    data = get_tagged_data()
    data_dict = {}

    for item in tqdm(data, desc="Processing tagged data"):
        all_pos = item['pos']
        filtered_pos = [p for p in all_pos if (not includes_pos or p['pos'] in includes_pos) and (
            not excludes_pos or p['pos'] not in excludes_pos)]
        data_dict[item['text']] = filtered_pos

    return data_dict


def get_normalized_words_counts(unigrams_pos_word_file, count_threshold=2, includes_pos=None, excludes_pos=None) -> list[str]:
    file_path = unigrams_pos_word_file
    data = load_data(file_path)
    word_details_dict = {}

    # Step 1: Collect counts and POS for each word
    for pos, pos_items in data.items():
        for word, count in pos_items.items():
            if pos != "PROPN" and word[0].isupper():
                continue

            lower_word = word.lower()
            if lower_word.isalpha():
                if lower_word not in word_details_dict:
                    word_details_dict[lower_word] = {pos: count}
                else:
                    if pos in word_details_dict[lower_word]:
                        word_details_dict[lower_word][pos] += count
                    else:
                        word_details_dict[lower_word][pos] = count

    # Step 2: Normalize by selecting the POS with the highest count and summing counts
    normalized_words = {}
    for word, details in word_details_dict.items():
        highest_pos = max(details, key=details.get)
        total_count = sum(details.values())
        if total_count >= count_threshold:
            normalized_words[word] = (highest_pos, total_count)

    # Step 3: Filter by includes_pos and excludes_pos
    normalized_words = {word: {"pos": pos, "count": count} for word,
                        (pos, count) in normalized_words.items()
                        if (not includes_pos or pos in includes_pos)
                        and (not excludes_pos or pos not in excludes_pos)}

    return normalized_words


@time_it
def get_english_words(count_threshold=10, includes_pos=None, excludes_pos=None) -> list[str]:
    en_dictionary = get_en_dictionary()
    en_words = list(en_dictionary.keys())
    return en_words


def get_english_word_info_dict(count_threshold=10, includes_pos=None, excludes_pos=None) -> list[str]:
    file_path = ENGLISH_UNIGRAMS_POS_WORD_FILE
    normalized_words = get_normalized_words_counts(
        file_path, count_threshold, includes_pos, excludes_pos)

    return normalized_words


@time_it
def get_tagalog_words(count_threshold=2, includes_pos=None, excludes_pos=None) -> list[str]:
    tl_dictionary = get_tl_dictionary(min_count=count_threshold,
                                      includes_pos=includes_pos, excludes_pos=excludes_pos)
    tl_words = list(tl_dictionary.keys())
    return tl_words


def get_tagalog_word_info_dict(count_threshold=2, includes_pos=None, excludes_pos=None) -> list[str]:
    file_path = TAGALOG_UNIGRAMS_POS_WORD_FILE
    normalized_words = get_normalized_words_counts(
        file_path, count_threshold, includes_pos, excludes_pos)

    return normalized_words


@time_it
def get_affixed_tagalog_words(*affixes):
    if not affixes:
        affixes = ["prefix", "infix", "suffix"]
    all_lemmas_affixed_dict = get_taggalog_affixed_info(*affixes)
    all_lemmas_affixed = list(all_lemmas_affixed_dict.keys())
    return all_lemmas_affixed


@time_it
def get_taggalog_affixed_info(*affixes):
    if not affixes:
        affixes = ["prefix", "infix", "suffix"]
    affixed_words_file = TAGALOG_AFFIXED_WORDS_FILE
    all_lemmas = load_data(affixed_words_file)
    all_lemmas_affixed_dict = {}
    all_items_affixed = []
    for affix in affixes:
        all_items_affixed.extend(all_lemmas.get(affix, {}).items())

    for _, item in tqdm(all_items_affixed, desc="Filtering affixed words"):
        word_counts = item['words'].items()
        for word, count in word_counts:
            if word[0].isupper():
                continue
            normalized_word = normalize_text(word)
            word_count = all_lemmas_affixed_dict.get(normalized_word, 0)
            if count > word_count:
                all_lemmas_affixed_dict[normalized_word] = count

    return all_lemmas_affixed_dict


@time_it
def get_tagalog_word_texts():
    files_dir = [
        "server/static/models/dost-asti-gpt2/base_model/datasets/base/tagalog_corpus_results/balita_nlp",
        "server/static/models/dost-asti-gpt2/base_model/datasets/base/tagalog_corpus_results/tl_unified"
    ]
    data = load_data_from_directories(files_dir, includes=["*_results.json"])
    texts = [text for item in data for text in item["texts"]]
    return texts


@time_it
def filter_tagalog_common_words(texts: list[str], min_word_count=5, includes_pos=None, excludes_pos=None):
    tagalog_words_info_dict = get_tagalog_word_info_dict(
        count_threshold=min_word_count, includes_pos=includes_pos, excludes_pos=excludes_pos)
    filtered_texts = []
    for text in texts:
        words = get_words(text)
        # Check if all words are in the words info dict
        if all(word in tagalog_words_info_dict for word in words):
            filtered_texts.append(text)
    return filtered_texts


def contains_tagalog_common_words(text: str, min_word_count=5, includes_pos=None, excludes_pos=None, threshold_percentage: float = 0.9) -> bool:
    tagalog_words_info_dict = get_tagalog_word_info_dict(
        count_threshold=min_word_count, includes_pos=includes_pos, excludes_pos=excludes_pos)
    words = get_words(text)
    tagalog_word_count = sum(
        1 for word in words if word in tagalog_words_info_dict)
    total_word_count = len(words)
    threshold_count = total_word_count * threshold_percentage
    return tagalog_word_count >= threshold_count


@time_it
def filter_english_common_words(texts: list[str], min_word_count=5, includes_pos=None, excludes_pos=None):
    english_words_info_dict = get_english_word_info_dict(
        count_threshold=min_word_count, includes_pos=includes_pos, excludes_pos=excludes_pos)
    filtered_texts = []
    for text in texts:
        words = get_words(text)
        # Check if all words are in the words info dict
        if all(word in english_words_info_dict for word in words):
            filtered_texts.append(text)
    return filtered_texts


def contains_english_common_words(text: str, min_word_count=5, includes_pos=None, excludes_pos=None, threshold_percentage: float = 0.9) -> bool:
    english_words_info_dict = get_english_word_info_dict(
        count_threshold=min_word_count, includes_pos=includes_pos, excludes_pos=excludes_pos)
    words = get_words(text)
    english_word_count = sum(
        1 for word in words if word in english_words_info_dict)
    total_word_count = len(words)
    threshold_count = total_word_count * threshold_percentage
    return english_word_count >= threshold_count


@time_it
def get_tagalog_md_texts(base_dir=None):
    tagalog_texts_dict = {}
    files_dir = base_dir or "data/scrapers/crawlers/datasets/"

    # Walk through all files in the directory
    for root, dirs, files in os.walk(files_dir):
        sorted_files = sorted(files)

        for file in sorted_files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                text = load_data(file_path)

                tagalog_texts_dict[file_path] = text

    return tagalog_texts_dict


@time_it
def get_proper_nouns():
    file_path = ENGLISH_UNIGRAMS_POS_WORD_FILE
    data = load_data(file_path)
    filtered_dict = {}
    min_count = 10

    for pos, pos_items in data.items():
        if pos == "PROPN":
            for word, count in pos_items.items():
                if count < min_count:
                    continue

                lower_word = word.lower()
                word_count = filtered_dict.get(lower_word, 0)
                if count > word_count:
                    filtered_dict[lower_word] = count

    # Sort by count in descending order
    filtered_dict = dict(sorted(filtered_dict.items(),
                         key=lambda item: item[1], reverse=True))
    return list(filtered_dict.keys())


@time_it
def get_word_pos_dict(includes_pos=None, excludes_pos=None):
    data = get_tagged_data()
    data_dict = {}

    for item in tqdm(data, desc="Processing tagged data"):
        all_pos = item['pos']
        # filtered_pos = [p for p in all_pos if (not includes_pos or p['pos'] in includes_pos) and (
        #     not excludes_pos or p['pos'] not in excludes_pos)]
        for pos in all_pos:
            word = pos['word']

            if (not includes_pos or pos['pos'] in includes_pos) and (not excludes_pos or pos['pos'] not in excludes_pos):
                if word not in data_dict:
                    data_dict[word] = pos['pos']

    return data_dict


def get_word_pos_list(selected_pos):
    word_pos_dict = get_word_pos_dict()
    words = [word for word, pos in word_pos_dict.items()if pos == selected_pos]
    return words


@time_it
def load_tagged_data_by_texts(data, lang, tagged_data: Optional[List] = None):
    tagged_data = tagged_data or get_tagged_data()
    tagged_data_dict = {item['text']: {**item, "lang": lang}
                        for item in tagged_data}

    filtered_tagged_data = []
    for item in tqdm(data, desc="Loading tagged data by texts"):
        if isinstance(item, str) and item in tagged_data_dict:
            filtered_tagged_data.append(tagged_data_dict[item])
        elif isinstance(item, dict):
            for key in item.keys():
                text = item[key]
                if isinstance(text, str) and text in tagged_data_dict:
                    filtered_tagged_data.append(tagged_data_dict[text])

    return filtered_tagged_data


def text_lambda(item):
    return item['text']


def has_matching_key(item, text):
    matches = False

    # Check if text is list
    if isinstance(text, list):
        text = '\n'.join(text)

    for key in item.keys():
        if isinstance(item, dict) and isinstance(item[key], str):
            text_to_match = item[key]

            # Update comparison to use whole word matching
            pattern = r'\b' + re.escape(text_to_match) + r'\b'
            print(f"text_to_match: {text_to_match}")
            if re.search(pattern, text):
                matches = True
                break

    return matches
