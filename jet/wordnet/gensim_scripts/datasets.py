from instruction_generator.helpers.words import count_words
import os
import json
import fnmatch
from tqdm import tqdm


def load_data_from_directories(source_directories, includes=None, excludes=None):
    data = []

    for directory in source_directories:
        # Check if directory is a json file
        if os.path.isfile(directory) and directory.endswith(".json"):
            source_file = directory
            with open(source_file, 'r') as file:
                data.extend(json.load(file))
            continue
        for filename in os.listdir(directory):
            # Apply include and exclude filters
            if (not includes or any(fnmatch.fnmatch(filename, pattern) for pattern in includes)) and \
               (not excludes or not any(fnmatch.fnmatch(filename, pattern) for pattern in excludes)):
                if filename.endswith(".json"):
                    source_file = os.path.join(directory, filename)

                    with open(source_file, 'r') as file:
                        data.extend(json.load(file))

    return data


def get_translation_sentences():
    directories = ["server/static/datasets/translations"]
    includes = ["jmbt*.json", "diwa*.json", "filwords*.json", "Ramos*.json"]
    excludes = []

    data = load_data_from_directories(directories, includes, excludes)
    sentences = []

    for item in data:
        # sentences.append(item['translation.tl'].lower())
        # sentences.append(item['translation.en'].lower())
        sentences.append(item['translation.tl'])
        sentences.append(item['translation.en'])

    return sentences


def get_translation_word_synonyms() -> dict:
    tl_file = "nanoGPT-LoRA/scripts/jet_bilingual/tagalog_enriched_dictionary.json"
    en_file = "nanoGPT-LoRA/scripts/jet_bilingual/english_enriched_dictionary.json"

    with open(tl_file, 'r', encoding='utf-8') as f:
        tl_data = json.load(f)
    with open(en_file, 'r', encoding='utf-8') as f:
        en_data = json.load(f)

    all_data = tl_data + en_data

    synonyms = {}

    for item in tqdm(all_data, desc="Getting base synonyms"):
        base_word = item['word']
        # less_freq_words = item.get("less_frequent_translations", [])
        top_translations = [translation_item['translation']
                            for translation_item in item.get("top_translations", [])]

        # all_words = less_freq_words + top_translations
        all_words = top_translations

        # Lowercase all words
        all_words = [word.lower() for word in all_words]

        # Filter out non-alpha words and words with more than one word
        all_words = [word for word in all_words if word.isalpha()
                     and count_words(word) == 1]

        synonyms[base_word] = all_words

    synonym_base = list(synonyms.keys())
    for synonym_word in tqdm(synonym_base, desc="Expanding synonyms"):
        base_list = synonyms[synonym_word]

        for word in base_list:
            if word in synonyms:
                all_synonyms = synonyms[synonym_word] + synonyms[word]
                # Sort by most frequent
                all_synonyms = sorted(
                    all_synonyms, key=lambda x: all_synonyms.count(x), reverse=True)
                synonyms[synonym_word] = all_synonyms

    # Remove duplicates and base word
    for synonym_word in tqdm(synonym_base, desc="Removing duplicates"):
        # Remove base word
        all_synonyms = [
            word for word in synonyms[synonym_word] if word != synonym_word]
        # Remove duplicates
        all_synonyms = list(set(all_synonyms))
        synonyms[synonym_word] = all_synonyms

        other_synonyms = []
        for word in all_synonyms:
            synonyms[word] = list(set(synonyms.get(word, []) + [synonym_word]))

            other_synonyms.extend(synonyms[word])

        all_synonyms = list(set(other_synonyms + all_synonyms))
        # Get index of base word
        index = all_synonyms.index(
            synonym_word) if synonym_word in all_synonyms else -1
        # Remove base word
        all_synonyms.pop(index) if index != -1 else None
        synonyms[synonym_word] = all_synonyms

    return synonyms


def get_translation_word_pairs(direction='en-tl') -> list:
    tl_file = "nanoGPT-LoRA/scripts/jet_bilingual/tagalog_enriched_dictionary.json"
    en_file = "nanoGPT-LoRA/scripts/jet_bilingual/english_enriched_dictionary.json"

    with open(tl_file, 'r', encoding='utf-8') as f:
        tl_data = json.load(f)
    with open(en_file, 'r', encoding='utf-8') as f:
        en_data = json.load(f)

    def get_word_pairs(data):
        word_pairs = []

        for item in data:
            base_word = item['word']

            # Check if base_word is a alpha word or has more than one word
            if not base_word.isalpha() or count_words(base_word) > 1:
                continue

            less_freq_words = item.get("less_frequent_translations", [])
            for word in less_freq_words:
                if not word.isalpha() or count_words(word) > 1:
                    continue

                word_pairs.append((base_word, word))
            top_translations = item.get("top_translations", [])
            for translation_item in top_translations:
                word = translation_item['translation']

                if not word.isalpha() or count_words(word) > 1:
                    continue

                word_pairs.append((base_word, word))

        return word_pairs

    tl_word_pairs = get_word_pairs(tl_data)
    en_word_pairs = get_word_pairs(en_data)

    word_pairs = []

    if direction == 'en-tl':
        word_pairs.extend(en_word_pairs)
        tl_word_pairs = [(tl, en) for en, tl in tl_word_pairs]
        word_pairs.extend(tl_word_pairs)
    else:
        word_pairs.extend(tl_word_pairs)
        en_word_pairs = [(en, tl) for tl, en in en_word_pairs]
        word_pairs.extend(en_word_pairs)

    return word_pairs
