import numpy as np
import os
from gensim.models import KeyedVectors
from instruction_generator.helpers.dataset import load_data, save_data
from instruction_generator.analyzers.helpers import get_tagalog_word_info_dict, get_english_word_info_dict
from instruction_generator.utils.time_utils import time_it
from instruction_generator.wordnet.embeddings.words_closest import WordSimilarityCalculator
from instruction_generator.wordnet.pos_tagger import POSTagger
from instruction_generator.train.preparation.model_sentence_evaluator import validate_sentence_pos
from tqdm import tqdm
from collections import defaultdict
from instruction_generator.wordnet.constants import TRANSLATION_FILES


def create_tl_en_words_mapping(word_translations_file, word_files_output_file, words_mapping=None, word_files_dict=None):
    includes_pos = ["NOUN", "VERB", "ADJ"]
    tagger = POSTagger()
    tl_word_info_dict = get_tagalog_word_info_dict()
    en_word_info_dict = get_english_word_info_dict()

    words_mapping = {pos: {}
                     for pos in includes_pos} if words_mapping is None else words_mapping
    # Updated to use defaultdict
    tl_en_word_files_dict = defaultdict(lambda: defaultdict(
        list)) if word_files_dict is None else word_files_dict
    count = 0
    batch_size = 30
    for file in TRANSLATION_FILES:
        data = load_data(file)
        basename = os.path.basename(file)

        pbar = tqdm(data, desc=f"File: {basename}")
        for idx, d in enumerate(data):
            pbar.update(1)

            tl_text = d["translation.tl"].strip().lower()
            en_text = d["translation.en"].strip().lower()

            tl_pos_results = tagger.filter_pos(
                tl_text, includes_pos, lang='tl')
            en_pos_results = tagger.filter_pos(
                en_text, includes_pos, lang='en')

            for pos in includes_pos:
                words_mapping_pos = words_mapping[pos]
                try:
                    tl_pos = [tl_word_info_dict[pos_result['word']]['pos']
                              for pos_result in tl_pos_results
                              if pos_result['pos'] == pos]
                    en_pos = [en_word_info_dict[pos_result['word']]['pos']
                              for pos_result in en_pos_results
                              if pos_result['pos'] == pos]
                    if not tl_pos or not en_pos or tl_pos != en_pos:  # Validate if the POS tags are the same
                        continue
                except Exception as e:
                    continue
                tl_text = " ".join([pos_result['word']
                                    for pos_result in tl_pos_results
                                    if pos_result['pos'] == pos])
                en_text = " ".join([pos_result['word']
                                    for pos_result in en_pos_results
                                    if pos_result['pos'] == pos])

                words_key = words_mapping_pos.get(tl_text, dict())
                if en_text not in words_key:
                    words_key.update({en_text: 1})
                else:
                    words_key[en_text] += 1
                words_mapping_pos[tl_text] = words_key

                # Update tl_en_word_files_dict dictionary
                tl_words = tl_text.split()
                for word in tl_words:
                    word_files_dict = tl_en_word_files_dict.get(word, dict())
                    en_words = en_text.split()
                    for en_word in en_words:
                        en_word_files = word_files_dict.get(en_word, list())
                        if basename not in en_word_files:
                            en_word_files.append(basename)
                        word_files_dict[en_word] = en_word_files
                    tl_en_word_files_dict[word] = word_files_dict

                words_mapping[pos] = words_mapping_pos

            count += 1
            if count % batch_size == 0 or idx == len(data) - 1:
                save_data(word_translations_file, words_mapping, write=True)
                save_data(word_files_output_file,
                          tl_en_word_files_dict, write=True)

            pbar.set_description_str(f"File: {basename}, Count: {count}")
    return words_mapping


@time_it
def normalize_word_translations(data_pos_dict, min_occurrences=3):
    data_dict = {base_text: translation_count_dict for pos, pos_dict in data_pos_dict.items()
                 for base_text, translation_count_dict in pos_dict.items()}

    one_word_keys_dict = {}
    for base_text, translation_count_dict in tqdm(data_dict.items(), desc="Normalizing one word translations"):
        base_words = base_text.split()
        if len(base_words) == 1:
            base_word = base_words[0]
            one_word_key_dict = one_word_keys_dict.get(
                base_word, dict())
            for translation_word, count in translation_count_dict.items():
                one_word_key_count = one_word_key_dict.get(
                    translation_word, 0)
                one_word_key_dict[translation_word] = int(
                    one_word_key_count) + int(count)
            one_word_keys_dict[base_word] = one_word_key_dict

    for base_text, translation_count_dict in tqdm(data_dict.items(), desc="Normalizing multi word translations"):
        base_words = base_text.split()
        if len(base_words) > 1:
            for base_word in base_words:
                # if base_word in one_word_keys_dict:
                #     continue
                for translation_text, count in translation_count_dict.items():
                    translation_words = translation_text.split()
                    translation_word_count_dict = one_word_keys_dict.get(
                        base_word, dict())
                    for translation_word in translation_words:
                        if translation_word not in translation_word_count_dict:
                            continue
                        translation_one_word_key_count = translation_word_count_dict.get(
                            translation_word, 0)
                        translation_word_count_dict[translation_word] = int(
                            translation_one_word_key_count) + int(count)
                    one_word_keys_dict[base_word] = translation_word_count_dict

    # Filter out translations with < min_occurrences
    one_word_keys_dict = {base_word: {translation_word: count for translation_word, count in translation_count_dict.items(
    ) if count >= min_occurrences} for base_word, translation_count_dict in one_word_keys_dict.items()}
    # Remove empty translation dictionaries
    one_word_keys_dict = {base_word: translation_count_dict for base_word,
                          translation_count_dict in one_word_keys_dict.items() if translation_count_dict}
    # Sort the dictionary by the number of occurrences in descending order
    one_word_keys_dict = {base_word: {translation_word: count for translation_word, count in sorted(
        translation_count_dict.items(), key=lambda item: item[1], reverse=True)} for base_word, translation_count_dict in one_word_keys_dict.items()}
    return one_word_keys_dict


@time_it
def format_embeddings_dataset(dataset):
    word_files_file = 'server/static/models/dost-asti-gpt2/base_model/datasets/base/word_texts/tl_en_word_files.json'
    word_files_dict = load_data(word_files_file)
    min_files = 3

    word_translations = list()
    for tagalog_word, translations in dataset.items():
        # Skip words with less than 3 files
        if len(word_files_dict.get(tagalog_word, [])) < min_files:
            continue
        # Add the Tagalog word/POS pair to the set
        word_translations.append(tagalog_word)
        for translation in translations:
            word_translations.append(translation)
    return word_translations


@time_it
def create_or_load_embeddings(dataset, embedding_file, vector_size=100):
    word_translations = format_embeddings_dataset(dataset)

    if os.path.exists(embedding_file):
        embeddings = KeyedVectors.load_word2vec_format(embedding_file)
    else:
        embeddings = {}
        for word in word_translations:
            # Initialize a random vector for each unique word/POS pair
            embeddings[word] = np.random.rand(vector_size)

        # Save the embeddings to a file in word2vec format
        with open(embedding_file, 'w') as f:
            # Write the header: number of word/POS pairs and vector size
            f.write(f"{len(embeddings)} {vector_size}\n")
            for word, vec in embeddings.items():
                # Write each word/POS pair and its vector
                f.write(f"{word} {' '.join(map(str, vec))}\n")

    # Load the embeddings with KeyedVectors for consistency
    return KeyedVectors.load_word2vec_format(embedding_file, binary=False)


if __name__ == '__main__':
    words_mapping_file = 'server/static/models/dost-asti-gpt2/base_model/datasets/base/word_texts/tl_en_words_mapping.json'
    word_files_output_file = 'server/static/models/dost-asti-gpt2/base_model/datasets/base/word_texts/tl_en_word_files.json'
    existing_words_mapping = load_data(words_mapping_file) if os.path.exists(
        words_mapping_file) else None
    existing_word_files_dict = load_data(word_files_output_file) if os.path.exists(
        word_files_output_file) else None
    words_mapping = create_tl_en_words_mapping(
        words_mapping_file, word_files_output_file, words_mapping=existing_words_mapping, word_files_dict=existing_word_files_dict)

    normalized_output_file = 'server/static/models/dost-asti-gpt2/base_model/datasets/base/word_texts/tl_en_word_normalized.json'
    one_word_keys_dict = normalize_word_translations(
        data_pos_dict=words_mapping)
    save_data(normalized_output_file, one_word_keys_dict, write=True)

    # word_translations = 'server/static/models/dost-asti-gpt2/base_model/datasets/base/word_texts/tl_en_word_normalized.json'
    # vector_size = 100
    # filename = f"tl_en_word2vec_{vector_size}.bin"
    # embedding_file = f"instruction_generator/wordnet/embeddings/{filename}"
    # dataset = load_data(word_translations)
    # embeddings = create_or_load_embeddings(
    #     dataset, embedding_file, vector_size=vector_size)
    # print(f"Embeddings count: {len(embeddings)}")

    # # Calculate the closest synonyms for each word in the dataset
    # calculator = WordSimilarityCalculator(embedding_file)
    # closest_synonyms = calculator.find_closest_synonyms(dataset)

    # for tagalog_word_pos, english_word_pos in closest_synonyms.items():
    #     print(f"{tagalog_word_pos}: {english_word_pos}")
