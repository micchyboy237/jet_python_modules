import pandas as pd
import json
import random
import re
from itertools import islice, tee, combinations
from instruction_generator.helpers.dataset import load_data_from_directories, save_data
from instruction_generator.helpers.words import get_words
from instruction_generator.helpers.string_sorter import StringSorter
from instruction_generator.analyzers.helpers import get_tagged_data, load_tagged_data_by_texts, text_lambda
from collections import defaultdict
from tqdm import tqdm
from instruction_generator.utils.time_utils import time_it
from typing import Dict, Any


class LanguageDataProcessor:
    def __init__(self, data, n=1, from_start=None, format_lambda=text_lambda):
        self.data = data
        self.from_start = from_start
        self.pos_word_counts = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int)))
        self.grouped_data = self.group_by_language(
            data, n, from_start, format_lambda)
        self.languages = list(self.grouped_data.keys())

    @time_it
    def group_by_language(self, tagged_data, n, from_start, format_lambda):
        grouped_data = {}

        for item in tqdm(tagged_data, desc="Grouping by language"):
            lang = item['lang']
            if lang not in grouped_data:
                grouped_data[lang] = {'texts': [], 'pos': []}

            text = ''
            pos_n_items = [{'word': '[BOS]', 'pos': '[BOS]'}]  # Prepend [BOS]

            if from_start:
                pos_n_items += item['pos'][:n] if from_start else item['pos']
                original_text = format_lambda(item)

                for pos in pos_n_items:
                    word = pos['word']
                    if word not in ['[BOS]', '[EOS]']:
                        word_index = original_text.find(word)
                        text += original_text[word_index:word_index +
                                              len(word)]
                        if word_index + len(word) < len(original_text) and original_text[word_index + len(word)] == ' ':
                            text += ' '
                text = text.strip()
            else:
                text = format_lambda(item)
                pos_n_items += item['pos']

            pos_n_items.append(
                {'word': '[EOS]', 'pos': '[EOS]'})  # Append [EOS]

            grouped_data[lang]['texts'].append(text)
            grouped_data[lang]['pos'].extend(pos_n_items)

        return grouped_data

    @time_it
    def nwise(self, iterable, n=2):
        "Returns a sliding window (of width n) over data from the iterable"
        iters = tee(iterable, n)
        for i, it in enumerate(iters):
            next(islice(it, i, i), None)
        return zip(*iters)

    @time_it
    def pos_combinations(self, pos_list, n=2):
        "Returns all possible combinations of POS of length n"
        return combinations(pos_list, n)

    def process_language_data(self, n=2, top_n=5, all_combinations=False):
        results = {}
        for lang, data in tqdm(self.grouped_data.items(), desc="Processing language data"):
            df = pd.DataFrame(data['pos'])
            df['word_length'] = df['word'].apply(len)
            average_word_length = df['word_length'].mean()

            word_counts = [len(text.split()) for text in data['texts']]
            average_word_count = sum(word_counts) / len(word_counts)

            pos_counts = df['pos'].value_counts()

            top_most_common_pos = pos_counts.head(top_n).to_dict()
            top_least_common_pos = pos_counts.tail(top_n).to_dict()
            # Convert to serialized string
            top_most_common_pos = {
                str(k): v for k, v in top_most_common_pos.items()}
            top_least_common_pos = {
                str(k): v for k, v in top_least_common_pos.items()}

            results[lang] = {
                "average_word_length": average_word_length,
                "average_word_count": average_word_count,
                "top_most_common_pos": top_most_common_pos,
                "top_least_common_pos": top_least_common_pos
            }

            pos_sequence_counts = None
            if all_combinations:
                pos_sequence_counts = pd.Series(
                    list(self.pos_combinations(df['pos'], n))).value_counts()
            else:
                pos_sequence_counts = pd.Series(
                    list(self.nwise(df['pos'], n))).value_counts()

            top_most_common_pos_sequences = pos_sequence_counts.head(
                top_n).to_dict()
            top_least_common_pos_sequences = pos_sequence_counts.tail(
                top_n).to_dict()
            # Convert to serialized string
            top_most_common_pos_sequences = {
                str(k): v for k, v in top_most_common_pos_sequences.items()}
            top_least_common_pos_sequences = {
                str(k): v for k, v in top_least_common_pos_sequences.items()}

            results[lang].update({
                "top_most_common_pos_sequences": top_most_common_pos_sequences,
                "top_least_common_pos_sequences": top_least_common_pos_sequences
            })

            # N-grams analysis
            ngrams_list = []
            for text in data['texts']:
                ngrams_list.extend(get_words(text, n))

            ngram_counts = pd.Series(ngrams_list).value_counts()

            top_most_common_ngrams = ngram_counts.head(top_n).to_dict()
            top_least_common_ngrams = ngram_counts.tail(top_n).to_dict()
            # Convert to serialized string
            top_most_common_ngrams = {
                str(k): v for k, v in top_most_common_ngrams.items()}
            top_least_common_ngrams = {
                str(k): v for k, v in top_least_common_ngrams.items()}

            results[lang].update({
                "top_most_common_ngrams": top_most_common_ngrams,
                "top_least_common_ngrams": top_least_common_ngrams
            })

        return results

    @time_it
    def get_pos_word_counts(self, n=1, includes_pos=None, excludes_pos=None):
        def generate_pos_word_sequences(pos_list, n):
            return [' '.join([f"{pos['word']}/{pos['pos']}" for pos in pos_seq])
                    for pos_seq in self.nwise(pos_list, n)]

        def filter_starting_sequences(sequences):
            filtered_sequences = sequences
            if self.from_start:
                filtered_sequences = [
                    seq for seq in sequences if seq.startswith("[BOS]")]

            return filtered_sequences

        def filter_sequences(sequences, includes, excludes):
            return [seq for seq in sequences if (not includes or any(pos in seq for pos in includes)) and
                    (not excludes or not any(pos in seq for pos in excludes))]

        sorted_pos_word_counts = {}
        for lang, data in tqdm(self.grouped_data.items(), desc="Generating POS word counts"):
            pos_word_n_sequences = generate_pos_word_sequences(
                data['pos'], n)
            pos_word_n_sequences = filter_starting_sequences(
                pos_word_n_sequences)
            filtered_sequences = filter_sequences(
                pos_word_n_sequences, includes_pos, excludes_pos)
            pos_word_sequence_counts = pd.Series(
                filtered_sequences).value_counts()
            sorted_pos_word_counts[lang] = {k: (v.item() if hasattr(v, 'item') else v)
                                            for k, v in pos_word_sequence_counts.items()}

        return sorted_pos_word_counts

    @time_it
    def get_pos_sequence_counts(self, n=1, includes_pos=None, excludes_pos=None):
        def generate_pos_sequences(pos_list, n):
            return [' '.join(pos_seq) for pos_seq in self.nwise(pos_list, n)]

        def filter_starting_sequences(sequences):
            filtered_sequences = sequences
            if self.from_start:
                filtered_sequences = [
                    seq for seq in sequences if seq.startswith("[BOS]")]

            return filtered_sequences

        def filter_sequences(sequences, includes, excludes):
            return [seq for seq in sequences if (not includes or any(pos in seq for pos in includes)) and
                    (not excludes or not any(pos in seq for pos in excludes))]

        sorted_pos_sequence_counts = {}
        for lang, data in tqdm(self.grouped_data.items(), desc="Generating POS sequence counts"):
            pos_n_sequences = generate_pos_sequences(
                [pos['pos'] for pos in data['pos']], n)
            pos_n_sequences = filter_starting_sequences(pos_n_sequences)
            filtered_sequences = filter_sequences(
                pos_n_sequences, includes_pos, excludes_pos)
            pos_sequence_counts = pd.Series(
                filtered_sequences).value_counts()
            sorted_pos_sequence_counts[lang] = {k: (v.item() if hasattr(v, 'item') else v)
                                                for k, v in pos_sequence_counts.items()}

        return sorted_pos_sequence_counts

    @time_it
    def generate_word_dict(self, pos_word_counts, n):
        results_dict = {}
        for lang, pos_word_counts in tqdm(pos_word_counts.items(), desc="Generating word dicts"):
            word_dict = {}
            for wordpos, word_counts in pos_word_counts.items():
                words_and_pos = wordpos.split()
                words = [w.split('/')[0] for w in words_and_pos if '/' in w]
                pos_tags = [w.split('/')[1] for w in words_and_pos if '/' in w]
                word_sequence = ' '.join(words)
                pos_sequence = ' '.join(pos_tags)

                if len(pos_sequence.strip().split(" ")) != n:
                    continue

                if not pos_sequence or not word_sequence:
                    continue

                if pos_sequence not in word_dict:
                    word_dict[pos_sequence] = {}

                word_dict[pos_sequence][word_sequence] = word_counts

            results_dict[lang] = word_dict
        return results_dict

    def generate_word_texts(self, word_dict, languages, pos_text_count=None):
        all_pos_texts_dict = {d['pos_text']: {
            'lang': d['lang'],
            'text': d['text']
        } for d in self.data}
        word_texts = {}

        for lang in languages:
            allowed_pos_sequences_per_tag = defaultdict(list)
            for pos_tag, word_dict in word_dict[lang].items():
                pos_tag = pos_tag.strip()

                if pos_tag in ['PART', 'PUNCT']:
                    continue

                words = list(word_dict.keys())
                for word in words:
                    if pos_tag not in allowed_pos_sequences_per_tag:
                        allowed_pos_sequences_per_tag[pos_tag] = []
                    allowed_pos_sequences_per_tag[pos_tag].append(word)

            # Shuffle and concatenate sequences from each POS tag
            allowed_pos_sequences = []
            for pos_sequences in allowed_pos_sequences_per_tag.values():
                allowed_pos_sequences.extend(pos_sequences[:pos_text_count])

            for pos_sequence in tqdm(allowed_pos_sequences, desc=f"Generating '{lang}' word texts"):
                # Find one of all_pos_texts_dict that contains the pos_sequence
                for pos_text, pos_dict in all_pos_texts_dict.items():
                    text = pos_dict['text']
                    lang = pos_dict['lang']

                    if lang not in word_texts:
                        word_texts[lang] = {}

                    all_texts = word_texts[lang].values()
                    # Update comparison to use whole word matching
                    pattern = r'\b' + re.escape(pos_sequence) + r'\b'
                    if pos_sequence not in word_texts[lang] and re.search(pattern, pos_text):
                        capitalized_text = text.capitalize()

                        # Check if capitalized_text is already in word_texts
                        if capitalized_text in all_texts:
                            continue

                        if pos_sequence not in word_texts[lang]:
                            # Initialize as array
                            word_texts[lang][pos_sequence] = []
                        elif pos_text_count and len(word_texts[lang][pos_sequence]) >= pos_text_count:
                            continue

                        word_texts[lang][pos_sequence].append(capitalized_text)
                        break

        # Format word_texts by language[POS][word] = text
        formatted_pos_texts = {}
        for lang, pos_texts in word_texts.items():
            if not lang in formatted_pos_texts:
                formatted_pos_texts[lang] = {}

            for pos_sequence, text in pos_texts.items():
                for word_sequence in pos_sequence.split(" "):
                    word = word_sequence.split("/")[0].lower()
                    pos = word_sequence.split("/")[1]

                    if not pos in formatted_pos_texts[lang]:
                        formatted_pos_texts[lang][pos] = {}

                    if word not in formatted_pos_texts[lang][pos]:
                        formatted_pos_texts[lang][pos][word] = text

            # Sort formatted_pos_texts[lang][pos] alphabetically
            for pos in formatted_pos_texts[lang]:
                sorted_words = dict(
                    sorted(formatted_pos_texts[lang][pos].items()))
                formatted_pos_texts[lang][pos] = sorted_words

            word_texts[lang] = formatted_pos_texts[lang]

        return word_texts

    @time_it
    def extract_data(self, n=1, min_n=1, max_n=None, includes_pos=None, excludes_pos=None):
        def generate_pos_text_sequences(text_list, n):
            return [' '.join([text for text in text_seq])
                    for text_seq in self.nwise(text_list, n)]

        def filter_sequences(sequences, includes, excludes, min_n, max_n):
            matched_word_count = {}
            global_word_count = {}

            # Count words in sequences
            for seq in sequences:
                for pos in includes:
                    if pos not in matched_word_count:
                        matched_word_count[pos] = {}
                        global_word_count[pos] = {}
                    matched_pos_words = re.findall(
                        rf'\b([\w,:-]+)/{pos}\b', seq)
                    for word in matched_pos_words:
                        if word not in matched_word_count[pos]:
                            matched_word_count[pos][word] = 0
                            global_word_count[pos][word] = 0
                        global_word_count[pos][word] += 1

            global_filtered_sequences = []

            for seq in sequences:
                # Filter out sequences with words that are < min_n
                include_seq = True
                for pos in includes:
                    matched_pos_words = re.findall(
                        rf'\b([\w,:-]+)/{pos}\b', seq)
                    # If any word pair is < min_n, exclude the sequence
                    if not matched_pos_words or any(global_word_count[pos][word] < min_n for word in matched_pos_words):
                        include_seq = False
                        break

                if include_seq:
                    global_filtered_sequences.append(seq)

            filtered_sequences = []
            removed_sequence_words = []

            for seq in global_filtered_sequences:
                include_seq = True
                if (not includes or any(f"/{pos}" in seq for pos in includes)) and (not excludes or not any(f"/{pos}" in seq for pos in excludes)):
                    word_pos_pairs = re.findall(
                        r'\b([\w,:-]+)/([\w,:-]+)\b', seq)

                    # Count word pos pairs
                    for item in word_pos_pairs:
                        word = item[0]
                        pos = item[1]

                        if pos in includes:
                            count = matched_word_count[pos][word]

                            if count > max_n:
                                include_seq = False

                if include_seq:
                    word_pos_pairs = re.findall(
                        r'\b([\w,:-]+)/([\w,:-]+)\b', seq)

                    # Increment word counts
                    for item in word_pos_pairs:
                        word = item[0]
                        pos = item[1]

                        if pos in includes:
                            matched_word_count[pos][word] += 1
                    filtered_sequences.append(seq)

            sorted_word_count_in_desc = {}
            for pos, words in matched_word_count.items():
                sorted_word_count_in_desc[pos] = dict(
                    sorted(words.items(), key=lambda item: item[1], reverse=True))
            # Filter words that are 0
            removed_sequence_words = {
                pos: [word for word, count in words.items() if count == 0] for pos, words in sorted_word_count_in_desc.items() if pos in includes}
            # Remove word counts that are 0
            sorted_word_count_in_desc = {
                pos: {word: count for word, count in words.items() if count > 0} for pos, words in sorted_word_count_in_desc.items()}

            return {
                'word_count': sorted_word_count_in_desc,
                'sequences': filtered_sequences,
                'removed_sequence_words': removed_sequence_words
            }

        result_dict = {}
        word_count = {}

        for lang, data in tqdm(self.grouped_data.items(), desc="Generating POS word counts"):
            pos_text_n_sequences = generate_pos_text_sequences(
                data['texts'], n)
            result_dict[lang] = filter_sequences(
                pos_text_n_sequences, includes_pos, excludes_pos, min_n, max_n)
            word_count = result_dict[lang]['word_count']
            removed_sequence_words = result_dict[lang]['removed_sequence_words']

        filtered_data = []
        pos_text_dict = {d['pos_text']: d for d in self.data}
        for lang, lang_data in result_dict.items():
            for pos_text in lang_data['sequences']:
                if pos_text in pos_text_dict:
                    filtered_data.append(pos_text_dict[pos_text])

        # Extract remaining data
        filtered_sequences = set(d['pos_text'] for d in filtered_data)
        remaining_data = [
            d for d in self.data if d['pos_text'] not in filtered_sequences]

        return {
            'data': filtered_data,
            'remaining_data': remaining_data,
            'info': {
                'word_count': word_count,
                'removed_sequence_words': removed_sequence_words,
            }
        }

    def generate_analysis(self, n=1, top_n=10, includes_pos=[], excludes_pos=[], words_only=False, word_texts=False, pos_text_count=None):
        languages = self.languages

        pos_word_counts = self.get_pos_word_counts(
            n, includes_pos, excludes_pos)

        yield {
            "type": "pos_word_counts",
            "languages": languages,
            "data": pos_word_counts
        }

        # return_dict = {
        #     'languages': languages,
        #     'pos_word_counts': pos_word_counts,
        # }

        if words_only or word_texts:
            word_dict = self.generate_word_dict(pos_word_counts, n)

            yield {
                "type": "word_dict",
                "languages": languages,
                "data": word_dict
            }

            # return_dict.update({
            #     'word_dict': word_dict,
            # })

            if word_texts:
                word_texts = self.generate_word_texts(
                    word_dict, languages, pos_text_count)

                yield {
                    "type": "word_texts",
                    "languages": languages,
                    "data": word_texts
                }

                # return_dict.update({
                #     'word_texts': word_texts
                # })
        else:
            pos_sequence_counts = self.get_pos_sequence_counts(
                n, includes_pos, excludes_pos)

            yield {
                "type": "pos_sequence_counts",
                "languages": languages,
                "data": pos_sequence_counts
            }

            # return_dict.update({
            #     'pos_sequence_counts': pos_sequence_counts
            # })

        language_results = self.process_language_data(n, top_n)

        yield {
            "type": "language_results",
            "languages": languages,
            "data": language_results
        }

        # return {
        #     **return_dict,
        #     'language_results': language_results
        # }


def clean_text(text: str) -> str:
    """Clean a text by removing [EOS], [BOS], and their variations with slashes."""
    cleaned_text = re.sub(r'\[EOS\]|\[BOS\]', '', text)
    cleaned_text = re.sub(r'/?\[EOS\]/?|/?\[BOS\]/?', '', cleaned_text)
    return cleaned_text.strip(' /')


def clean_dictionary(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively clean all keys and values in a dictionary, ensuring keys that are meaningful after cleaning are retained."""
    cleaned_dict = {}
    for key, value in data.items():
        cleaned_key = clean_text(key).strip()
        # Check if the key is meaningful after cleaning
        if cleaned_key or key.strip():
            if isinstance(value, dict):
                # Recurse into nested dictionaries
                cleaned_value = clean_dictionary(value)
            elif isinstance(value, str):
                cleaned_value = clean_text(value)
            else:
                cleaned_value = value  # Non-string, non-dict values are left as is
            cleaned_dict[cleaned_key if cleaned_key else key.strip()
                         ] = cleaned_value
    return cleaned_dict


def main(
    tagged_data,
    n=1,
    top_n=10,
    source_key='text',
    words_only=False,
    word_texts=False,
    from_start=None,
    all_combinations=False,
    includes_pos=[],
    excludes_pos=[],
    pos_text_count=None,
):
    processor = LanguageDataProcessor(
        tagged_data, n=n, from_start=from_start, format_lambda=text_lambda)

    # Generate analysis data
    result_stream = processor.generate_analysis(
        n=n,
        top_n=top_n,
        includes_pos=includes_pos,
        excludes_pos=excludes_pos,
        words_only=words_only,
        word_texts=word_texts,
        pos_text_count=pos_text_count,
    )

    n_label = n - 1 if from_start else n
    n_scope = "start" if from_start else "all"

    for result in result_stream:
        type = result['type']
        languages = result['languages']
        data = result['data']

        for lang in languages:
            result = clean_dictionary(data[lang])

            if type == "word_dict":
                save_data(
                    f'instruction_generator/analyzers/datasets/{lang}/{n_label}_word_dict_{n_scope}.json', result, write=True)
            elif type == "word_texts":
                save_data(
                    f'instruction_generator/analyzers/datasets/{lang}/{n_label}_word_texts_{n_scope}.json', result, write=True)
            elif type == "pos_word_counts":
                save_data(
                    f'instruction_generator/analyzers/datasets/{lang}/{n_label}_pos_word_counts_{n_scope}.json', result, write=True)
            elif type == "pos_sequence_counts":
                save_data(
                    f'instruction_generator/analyzers/datasets/{lang}/{n_label}_pos_sequence_counts_{n_scope}.json', result, write=True)
            elif type == "language_results":
                save_data(
                    f'instruction_generator/analyzers/datasets/{lang}/{n_label}_language_results_{n_scope}.json', result, write=True)


if __name__ == '__main__':
    tagged_data = get_tagged_data()

    # From Start
    main(tagged_data, n=2, from_start=True, words_only=True)
    main(tagged_data, n=3, from_start=True, words_only=True)

    # From All
    excludes_pos = ['[BOS]', '[EOS]']
    main(tagged_data, n=1, from_start=False, words_only=True)
    main(tagged_data, n=2, from_start=False, words_only=True)
    main(tagged_data, n=1, from_start=False, excludes_pos=excludes_pos)
    main(tagged_data, n=2, from_start=False, excludes_pos=excludes_pos)

    # excludes_pos = ['[BOS]', '[EOS]']
    # main(tagged_data, n=1, words_only=True,
    #      from_start=None, excludes_pos=excludes_pos, pos_text_count=100)

    # excludes_pos = ['CCONJ [EOS]', 'PUNCT [EOS]', '[EOS] [BOS]']
    # main(tagged_data, n=2, words_only=True,
    #      from_start=None, excludes_pos=excludes_pos)

    # excludes_pos = ['CCONJ [EOS]', 'PUNCT [EOS]', '[EOS] [BOS]']
    # main(tagged_data, n=2, from_start=None, excludes_pos=excludes_pos)

    # includes_pos = ["[BOS]"]
    # main(tagged_data, n=3, from_start=True,
    #      includes_pos=includes_pos, excludes_pos=excludes_pos)
