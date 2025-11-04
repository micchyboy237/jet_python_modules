import os
import pandas as pd
import re
from itertools import islice, tee, combinations
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Any, List, Set
from jet.file.utils import save_data
from jet.logger.timer import time_it
from jet.wordnet.analyzers.helpers import text_lambda
from jet.wordnet.words import get_words
from jet.wordnet.pos_tagger import POSTagger

try:
    from nltk.corpus import stopwords
    _STOPWORDS: Set[str] = set(stopwords.words('english'))
except Exception:  # pragma: no cover
    _STOPWORDS = set()  # fallback – no stop-word removal if NLTK not installed

class LanguageDataProcessor:
    def __init__(self, data: List[Dict[str, Any]], n=1, from_start=None, format_lambda=text_lambda):
        self.data = data
        self.from_start = from_start
        self.pos_word_counts = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int)))
        self.polymorphic_data = defaultdict(list)
        for item in data:
            # Default to 'en' if lang not specified
            item['lang'] = item.get('lang', 'en')
            self.polymorphic_data[item['lang']].append(item)
        self.grouped_data = self.group_by_language(
            data, n, from_start, format_lambda)
        self.languages = list(self.grouped_data.keys())

    @staticmethod
    def _filter_stopwords(words: List[str]) -> List[str]:
        """Remove English stop-words (lower-cased)."""
        return [w for w in words if w.lower() not in _STOPWORDS]

    @time_it
    def group_by_language(self, tagged_data, n, from_start, format_lambda):
        grouped_data = {}
        for item in tqdm(tagged_data, desc="Grouping by language"):
            lang = item['lang']
            if lang not in grouped_data:
                grouped_data[lang] = {'texts': [], 'pos': []}
            text = format_lambda(item)
            pos_n_items = item['pos']
            if from_start:
                pos_n_items = [
                    {'word': '[BOS]', 'pos': '[BOS]'}] + pos_n_items[:n]
            else:
                pos_n_items = [{'word': '[BOS]', 'pos': '[BOS]'}] + pos_n_items
            pos_n_items.append({'word': '[EOS]', 'pos': '[EOS]'})
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

    def process_language_data(
        self,
        n: int = 2,
        top_n: int = 5,
        all_combinations: bool = False,
        includes_pos: List[str] | None = None,
        excludes_pos: List[str] | None = None
    ) -> Dict[str, Any]:
        """
        Process grouped language data to compute:
        - Average word length and count
        - Top/least common POS tags
        - Top/least common POS sequences
        - Top/least common n-grams (filtered by POS if specified)
        """
        results = {}
        includes_pos = includes_pos or []
        excludes_pos = excludes_pos or []

        for lang, data in tqdm(self.grouped_data.items(), desc="Processing language data"):
            # === 1. Build and filter POS DataFrame ===
            df = pd.DataFrame(data['pos'])
            if includes_pos or excludes_pos:
                mask = pd.Series([True] * len(df), index=df.index)
                if includes_pos:
                    mask &= df['pos'].isin(includes_pos)
                if excludes_pos:
                    mask &= ~df['pos'].isin(excludes_pos)
                df = df[mask]  # Filter rows by POS

            # === 2. Basic statistics ===
            df['word_length'] = df['word'].apply(len)
            average_word_length = df['word_length'].mean()
            word_counts = [len(text.split()) for text in data['texts']]
            average_word_count = sum(word_counts) / len(word_counts) if word_counts else 0

            pos_counts = df['pos'].value_counts()
            top_most_common_pos = pos_counts.head(top_n).to_dict()
            top_least_common_pos = pos_counts.tail(top_n).to_dict()
            top_most_common_pos = {str(k): v for k, v in top_most_common_pos.items()}
            top_least_common_pos = {str(k): v for k, v in top_least_common_pos.items()}

            results[lang] = {
                "average_word_length": average_word_length,
                "average_word_count": average_word_count,
                "top_most_common_pos": top_most_common_pos,
                "top_least_common_pos": top_least_common_pos
            }

            # === 3. POS sequence counts (with filtering) ===
            pos_seq_iter = (
                self.pos_combinations(df['pos'], n)
                if all_combinations else
                self.nwise(df['pos'], n)
            )
            pos_sequence_counts = pd.Series(list(pos_seq_iter)).value_counts()

            if includes_pos or excludes_pos:
                def seq_matches(seq):
                    if includes_pos and not any(p in seq for p in includes_pos):
                        return False
                    if excludes_pos and any(p in seq for p in excludes_pos):
                        return False
                    return True
                pos_sequence_counts = pos_sequence_counts[pos_sequence_counts.index.map(seq_matches)]

            top_most_common_pos_sequences = pos_sequence_counts.head(top_n).to_dict()
            top_least_common_pos_sequences = pos_sequence_counts.tail(top_n).to_dict()
            top_most_common_pos_sequences = {str(k): v for k, v in top_most_common_pos_sequences.items()}
            top_least_common_pos_sequences = {str(k): v for k, v in top_least_common_pos_sequences.items()}

            results[lang].update({
                "top_most_common_pos_sequences": top_most_common_pos_sequences,
                "top_least_common_pos_sequences": top_least_common_pos_sequences
            })

            # === 4. N-gram counts with POS-aware filtering ===
            filtered_ngrams = []

            if includes_pos or excludes_pos:
                word_pos_map = dict(zip(df['word'], df['pos']))
                for text in data['texts']:
                    words = text.split()
                    # ---- stop-word removal for the whole text ----
                    words = self._filter_stopwords(words)

                    for ngram in get_words(' '.join(words), n):
                        ngram_words = ngram.split()
                        if len(ngram_words) != n:
                            continue

                        # locate n-gram in (already stop-word-filtered) text
                        start_idx = ' '.join(words).find(ngram)
                        if start_idx == -1:
                            continue
                        word_start = len(' '.join(words)[:start_idx].split())

                        pos_tags = []
                        for i in range(n):
                            idx = word_start + i
                            if idx < len(words) and words[idx] == ngram_words[i]:
                                pos = word_pos_map.get(words[idx])
                                if pos:
                                    pos_tags.append(pos)

                        if len(pos_tags) != n:
                            continue

                        # ---- strict POS inclusion ----
                        include = True
                        if includes_pos and not all(p in includes_pos for p in pos_tags):
                            include = False
                        if excludes_pos and any(p in excludes_pos for p in pos_tags):
                            include = False
                        if include:
                            filtered_ngrams.append(ngram)
            else:
                # no POS filter → still remove stop-words
                for text in data['texts']:
                    cleaned = ' '.join(self._filter_stopwords(text.split()))
                    filtered_ngrams.extend(get_words(cleaned, n))

            ngram_counts = pd.Series(filtered_ngrams).value_counts()
            top_most_common_ngrams = ngram_counts.head(top_n).to_dict()
            top_least_common_ngrams = ngram_counts.tail(top_n).to_dict()
            top_most_common_ngrams = {str(k): v for k, v in top_most_common_ngrams.items()}
            top_least_common_ngrams = {str(k): v for k, v in top_least_common_ngrams.items()}

            results[lang].update({
                "top_most_common_ngrams": top_most_common_ngrams,
                "top_least_common_ngrams": top_least_common_ngrams
            })

        return results

    @time_it
    def get_pos_word_counts(self, n=1, includes_pos=None, excludes_pos=None):
        includes_pos = includes_pos or []
        excludes_pos = excludes_pos or []

        def generate_pos_word_sequences(pos_list, n):
            return [
                ' '.join([f"{pos['word']}/{pos['pos']}" for pos in pos_seq])
                for pos_seq in self.nwise(pos_list, n)
            ]

        def filter_starting_sequences(sequences):
            if self.from_start:
                return [seq for seq in sequences if seq.startswith("[BOS]")]
            return sequences

        def filter_by_pos_and_stopwords(seq: str) -> bool:
            # POS filter
            pos_tags = [part.split('/')[1] for part in seq.split() if '/' in part]
            if includes_pos and not all(p in includes_pos for p in pos_tags):
                return False
            if excludes_pos and any(p in excludes_pos for p in pos_tags):
                return False
            # stop-word filter
            words = [part.split('/')[0] for part in seq.split() if '/' in part]
            if any(w.lower() in _STOPWORDS for w in words):
                return False
            return True

        sorted_pos_word_counts = {}
        for lang, data in tqdm(self.grouped_data.items(),
                               desc="Generating POS word counts"):
            sequences = generate_pos_word_sequences(data['pos'], n)
            sequences = filter_starting_sequences(sequences)
            sequences = [s for s in sequences if filter_by_pos_and_stopwords(s)]

            counts = pd.Series(sequences).value_counts()
            sorted_pos_word_counts[lang] = {
                k: (v.item() if hasattr(v, 'item') else v)
                for k, v in counts.items()
            }
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
            allowed_pos_sequences = []
            for pos_sequences in allowed_pos_sequences_per_tag.values():
                allowed_pos_sequences.extend(pos_sequences[:pos_text_count])
            for pos_sequence in tqdm(allowed_pos_sequences, desc=f"Generating '{lang}' word texts"):
                for pos_text, pos_dict in all_pos_texts_dict.items():
                    text = pos_dict['text']
                    lang = pos_dict['lang']
                    if lang not in word_texts:
                        word_texts[lang] = {}
                    all_texts = word_texts[lang].values()
                    pattern = r'\b' + re.escape(pos_sequence) + r'\b'
                    if pos_sequence not in word_texts[lang] and re.search(pattern, pos_text):
                        capitalized_text = text.capitalize()
                        if capitalized_text in all_texts:
                            continue
                        if pos_sequence not in word_texts[lang]:
                            word_texts[lang][pos_sequence] = []
                        elif pos_text_count and len(word_texts[lang][pos_sequence]) >= pos_text_count:
                            continue
                        word_texts[lang][pos_sequence].append(capitalized_text)
                        break
        formatted_pos_texts = {}
        for lang, pos_texts in word_texts.items():
            if lang not in formatted_pos_texts:
                formatted_pos_texts[lang] = {}
            for pos_sequence, text in pos_texts.items():
                for word_sequence in pos_sequence.split(" "):
                    word = word_sequence.split("/")[0].lower()
                    pos = word_sequence.split("/")[1]
                    if pos not in formatted_pos_texts[lang]:
                        formatted_pos_texts[lang][pos] = {}
                    if word not in formatted_pos_texts[lang][pos]:
                        formatted_pos_texts[lang][pos][word] = text
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
                include_seq = True
                for pos in includes:
                    matched_pos_words = re.findall(
                        rf'\b([\w,:-]+)/{pos}\b', seq)
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
            removed_sequence_words = {
                pos: [word for word, count in words.items() if count == 0] for pos, words in sorted_word_count_in_desc.items() if pos in includes}
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

    def generate_analysis(self, n=1, top_n=10, all_combinations=False, includes_pos=[], excludes_pos=[], words_only=False, word_texts=False, pos_text_count=None):
        languages = self.languages
        pos_word_counts = self.get_pos_word_counts(
            n, includes_pos, excludes_pos)
        yield {
            "type": "pos_word_counts",
            "languages": languages,
            "data": pos_word_counts
        }
        if words_only or word_texts:
            word_dict = self.generate_word_dict(pos_word_counts, n)
            yield {
                "type": "word_dict",
                "languages": languages,
                "data": word_dict
            }
            if word_texts:
                word_texts = self.generate_word_texts(
                    word_dict, languages, pos_text_count)
                yield {
                    "type": "word_texts",
                    "languages": languages,
                    "data": word_texts
                }
        else:
            pos_sequence_counts = self.get_pos_sequence_counts(
                n, includes_pos, excludes_pos)
            yield {
                "type": "pos_sequence_counts",
                "languages": languages,
                "data": pos_sequence_counts
            }
        language_results = self.process_language_data(n, top_n, all_combinations, includes_pos, excludes_pos)
        yield {
            "type": "language_results",
            "languages": languages,
            "data": language_results
        }


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
        if cleaned_key or key.strip():
            if isinstance(value, dict):
                cleaned_value = clean_dictionary(value)
            elif isinstance(value, str):
                cleaned_value = clean_text(value)
            else:
                cleaned_value = value
            cleaned_dict[cleaned_key if cleaned_key else key.strip()
                         ] = cleaned_value
    return cleaned_dict


def analyze_pos_tags(
    texts: List[Dict[str, str]],
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
    output_dir=None
):
    tagger = POSTagger()
    tagged_data = []
    for text_dict in texts:
        text = text_dict[source_key]
        lang = text_dict.get('lang', 'en')
        pos_results = tagger.process_and_tag(text)
        tagged_text = tagger.format_tags(pos_results)
        tagged_data.append({
            'lang': lang,
            'text': text,
            'pos': pos_results,
            'pos_text': tagged_text
        })

    output_dir = output_dir or os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    processor = LanguageDataProcessor(
        tagged_data, n=n, from_start=from_start, format_lambda=text_lambda)
    result_stream = processor.generate_analysis(
        n=n,
        top_n=top_n,
        all_combinations=all_combinations,
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
                    f'{output_dir}/datasets/{lang}/{n_label}_word_dict_{n_scope}.json', result, overwrite=True)
            elif type == "word_texts":
                save_data(
                    f'{output_dir}/datasets/{lang}/{n_label}_word_texts_{n_scope}.json', result, overwrite=True)
            elif type == "pos_word_counts":
                save_data(
                    f'{output_dir}/datasets/{lang}/{n_label}_pos_word_counts_{n_scope}.json', result, overwrite=True)
            elif type == "pos_sequence_counts":
                save_data(
                    f'{output_dir}/datasets/{lang}/{n_label}_pos_sequence_counts_{n_scope}.json', result, overwrite=True)
            elif type == "language_results":
                save_data(
                    f'{output_dir}/datasets/{lang}/{n_label}_language_results_{n_scope}.json', result, overwrite=True)
