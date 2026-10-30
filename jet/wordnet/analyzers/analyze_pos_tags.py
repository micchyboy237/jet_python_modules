import re
from collections import defaultdict
from itertools import combinations, islice, tee
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

import pandas as pd
from jet.logger.timer import time_it
from jet.wordnet.analyzers.helpers import text_lambda
from jet.wordnet.pos_tagger import POSTagger
from jet.wordnet.words import get_words
from rich.console import Console
from tqdm import tqdm

console = Console()

try:
    from nltk.corpus import stopwords

    _STOPWORDS: Set[str] = set(stopwords.words("english"))
except Exception:
    _STOPWORDS = set()


class POSWordCountResult(TypedDict):
    """Structure for POS word counts."""

    pos_word_counts: Dict[str, Dict[str, int]]


class WordDictResult(TypedDict):
    """Structure for word dictionary results."""

    word_dict: Dict[str, Dict[str, Dict[str, int]]]


class WordTextsResult(TypedDict):
    """Structure for word texts results."""

    word_texts: Dict[str, Dict[str, Dict[str, List[str]]]]


class POSSequenceCountResult(TypedDict):
    """Structure for POS sequence counts."""

    pos_sequence_counts: Dict[str, Dict[str, int]]


class LanguageMetricsResult(TypedDict):
    """Structure for language metrics."""

    average_word_length: float
    average_word_count: float
    top_most_common_pos: Dict[str, int]
    top_least_common_pos: Dict[str, int]
    top_most_common_pos_sequences: Dict[str, int]
    top_least_common_pos_sequences: Dict[str, int]
    top_most_common_ngrams: Dict[str, int]
    top_least_common_ngrams: Dict[str, int]


class LanguageResultsResult(TypedDict):
    """Structure for language results."""

    language_results: Dict[str, LanguageMetricsResult]


class AnalysisResult(TypedDict, total=False):
    """Complete analysis result structure."""

    pos_word_counts: Dict[str, Dict[str, int]]
    word_dict: Dict[str, Dict[str, Dict[str, int]]]
    word_texts: Dict[str, Dict[str, Dict[str, List[str]]]]
    pos_sequence_counts: Dict[str, Dict[str, int]]
    language_results: Dict[str, LanguageMetricsResult]


class MergedAnalysisResults(TypedDict):
    """Structure for merged analysis results with descriptive keys."""

    pos_word_counts_all: Dict[str, Dict[str, int]]
    word_dict_all: Dict[str, Dict[str, Dict[str, int]]]
    word_texts_all: Dict[str, Dict[str, Dict[str, List[str]]]]
    pos_sequence_counts_all: Dict[str, Dict[str, int]]
    language_results_all: Dict[str, LanguageMetricsResult]


class LanguageDataProcessor:
    def __init__(
        self,
        data: List[Dict[str, Any]],
        n: int = 1,
        format_lambda: Any = text_lambda,
    ):
        self.data = data
        self.pos_word_counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self.polymorphic_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in data:
            item["lang"] = item.get("lang", "en")
            self.polymorphic_data[item["lang"]].append(item)
        self.grouped_data = self.group_by_language(data, n, format_lambda)
        self.languages = list(self.grouped_data.keys())

    @staticmethod
    def _filter_stopwords(words: List[str]) -> List[str]:
        """Remove English stop-words (lower-cased)."""
        return [w for w in words if w.lower() not in _STOPWORDS]

    @time_it
    def group_by_language(
        self,
        tagged_data: List[Dict[str, Any]],
        n: int,
        format_lambda: Any,
    ) -> Dict[str, Dict[str, List]]:
        grouped_data: Dict[str, Dict[str, List]] = {}
        for item in tqdm(tagged_data, desc="Grouping by language"):
            lang = item["lang"]
            if lang not in grouped_data:
                grouped_data[lang] = {"texts": [], "pos": []}
            text = format_lambda(item)
            pos_n_items = item["pos"]
            # Always include full context with BOS/EOS markers for any-position analysis
            pos_n_items = [{"word": "[BOS]", "pos": "[BOS]"}] + pos_n_items
            pos_n_items.append({"word": "[EOS]", "pos": "[EOS]"})
            grouped_data[lang]["texts"].append(text)
            grouped_data[lang]["pos"].extend(pos_n_items)
        return grouped_data

    @time_it
    def nwise(self, iterable: List, n: int = 2) -> zip:
        "Returns a sliding window (of width n) over data from the iterable"
        iters = tee(iterable, n)
        for i, it in enumerate(iters):
            next(islice(it, i, i), None)
        return zip(*iters)

    @time_it
    def pos_combinations(self, pos_list: List, n: int = 2) -> combinations:
        "Returns all possible combinations of POS of length n"
        return combinations(pos_list, n)

    def process_language_data(
        self,
        n: int = 2,
        top_n: int = 5,
        all_combinations: bool = False,
        includes_pos: Optional[List[str]] = None,
        excludes_pos: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process grouped language data to compute:
        - Average word length and count
        - Top/least common POS tags
        - Top/least common POS sequences
        - Top/least common n-grams (filtered by POS if specified)
        """
        results: Dict[str, Any] = {}
        includes_pos = includes_pos or []
        excludes_pos = excludes_pos or []
        for lang, data in tqdm(
            self.grouped_data.items(), desc="Processing language data"
        ):
            df = pd.DataFrame(data["pos"])
            if includes_pos or excludes_pos:
                mask = pd.Series([True] * len(df), index=df.index)
                if includes_pos:
                    mask &= df["pos"].isin(includes_pos)
                if excludes_pos:
                    mask &= ~df["pos"].isin(excludes_pos)
                df = df[mask]
            df["word_length"] = df["word"].apply(len)
            average_word_length = df["word_length"].mean()
            word_counts = [len(text.split()) for text in data["texts"]]
            average_word_count = (
                sum(word_counts) / len(word_counts) if word_counts else 0
            )
            pos_counts = df["pos"].value_counts()
            top_most_common_pos = pos_counts.head(top_n).to_dict()
            top_least_common_pos = pos_counts.tail(top_n).to_dict()
            top_most_common_pos = {str(k): v for k, v in top_most_common_pos.items()}
            top_least_common_pos = {str(k): v for k, v in top_least_common_pos.items()}
            results[lang] = {
                "average_word_length": average_word_length,
                "average_word_count": average_word_count,
                "top_most_common_pos": top_most_common_pos,
                "top_least_common_pos": top_least_common_pos,
            }
            pos_seq_iter = (
                self.pos_combinations(df["pos"], n)
                if all_combinations
                else self.nwise(df["pos"], n)
            )
            pos_sequence_counts = pd.Series(list(pos_seq_iter)).value_counts()
            if includes_pos or excludes_pos:

                def seq_matches(seq: Tuple) -> bool:
                    if includes_pos and not any(p in seq for p in includes_pos):
                        return False
                    if excludes_pos and any(p in seq for p in excludes_pos):
                        return False
                    return True

                pos_sequence_counts = pos_sequence_counts[
                    pos_sequence_counts.index.map(seq_matches)
                ]
            top_most_common_pos_sequences = pos_sequence_counts.head(top_n).to_dict()
            top_least_common_pos_sequences = pos_sequence_counts.tail(top_n).to_dict()
            top_most_common_pos_sequences = {
                str(k): v for k, v in top_most_common_pos_sequences.items()
            }
            top_least_common_pos_sequences = {
                str(k): v for k, v in top_least_common_pos_sequences.items()
            }
            results[lang].update(
                {
                    "top_most_common_pos_sequences": top_most_common_pos_sequences,
                    "top_least_common_pos_sequences": top_least_common_pos_sequences,
                }
            )
            filtered_ngrams = []
            if includes_pos or excludes_pos:
                word_pos_map = dict(zip(df["word"], df["pos"]))
                for text in data["texts"]:
                    words = text.split()
                    words = self._filter_stopwords(words)
                    for ngram in get_words(" ".join(words), n):
                        ngram_words = ngram.split()
                        if len(ngram_words) != n:
                            continue
                        start_idx = " ".join(words).find(ngram)
                        if start_idx == -1:
                            continue
                        word_start = len(" ".join(words)[:start_idx].split())
                        pos_tags = []
                        for i in range(n):
                            idx = word_start + i
                            if idx < len(words) and words[idx] == ngram_words[i]:
                                pos = word_pos_map.get(words[idx])
                                if pos:
                                    pos_tags.append(pos)
                        if len(pos_tags) != n:
                            continue
                        include = True
                        if includes_pos and not all(
                            p in includes_pos for p in pos_tags
                        ):
                            include = False
                        if excludes_pos and any(p in excludes_pos for p in pos_tags):
                            include = False
                        if include:
                            filtered_ngrams.append(ngram)
            else:
                for text in data["texts"]:
                    cleaned = " ".join(self._filter_stopwords(text.split()))
                    filtered_ngrams.extend(get_words(cleaned, n))
            ngram_counts = pd.Series(filtered_ngrams).value_counts()
            top_most_common_ngrams = ngram_counts.head(top_n).to_dict()
            top_least_common_ngrams = ngram_counts.tail(top_n).to_dict()
            top_most_common_ngrams = {
                str(k): v for k, v in top_most_common_ngrams.items()
            }
            top_least_common_ngrams = {
                str(k): v for k, v in top_least_common_ngrams.items()
            }
            results[lang].update(
                {
                    "top_most_common_ngrams": top_most_common_ngrams,
                    "top_least_common_ngrams": top_least_common_ngrams,
                }
            )
        return results

    @time_it
    def get_pos_word_counts(
        self,
        n: int = 1,
        includes_pos: Optional[List[str]] = None,
        excludes_pos: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, int]]:
        includes_pos = includes_pos or []
        excludes_pos = excludes_pos or []

        def generate_pos_word_sequences(
            pos_list: List[Dict[str, str]], n: int
        ) -> List[str]:
            return [
                " ".join([f"{pos['word']}/{pos['pos']}" for pos in pos_seq])
                for pos_seq in self.nwise(pos_list, n)
            ]

        def filter_by_pos_and_stopwords(seq: str) -> bool:
            pos_tags = [part.split("/")[1] for part in seq.split() if "/" in part]
            if includes_pos and not all(p in includes_pos for p in pos_tags):
                return False
            if excludes_pos and any(p in excludes_pos for p in pos_tags):
                return False
            words = [part.split("/")[0] for part in seq.split() if "/" in part]
            if any(w.lower() in _STOPWORDS for w in words):
                return False
            return True

        sorted_pos_word_counts: Dict[str, Dict[str, int]] = {}
        for lang, data in tqdm(
            self.grouped_data.items(), desc="Generating POS word counts"
        ):
            sequences = generate_pos_word_sequences(data["pos"], n)
            # No start filtering anymore
            sequences = [s for s in sequences if filter_by_pos_and_stopwords(s)]
            counts = pd.Series(sequences).value_counts()
            sorted_pos_word_counts[lang] = {
                k: (v.item() if hasattr(v, "item") else v) for k, v in counts.items()
            }
        return sorted_pos_word_counts

    @time_it
    def get_pos_sequence_counts(
        self,
        n: int = 1,
        includes_pos: Optional[List[str]] = None,
        excludes_pos: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, int]]:
        def generate_pos_sequences(pos_list: List[str], n: int) -> List[str]:
            return [" ".join(pos_seq) for pos_seq in self.nwise(pos_list, n)]

        def filter_sequences(
            sequences: List[str],
            includes: Optional[List[str]],
            excludes: Optional[List[str]],
        ) -> List[str]:
            return [
                seq
                for seq in sequences
                if (not includes or any(pos in seq for pos in includes))
                and (not excludes or not any(pos in seq for pos in excludes))
            ]

        sorted_pos_sequence_counts: Dict[str, Dict[str, int]] = {}
        for lang, data in tqdm(
            self.grouped_data.items(), desc="Generating POS sequence counts"
        ):
            pos_n_sequences = generate_pos_sequences(
                [pos["pos"] for pos in data["pos"]], n
            )
            # No start filtering anymore
            filtered_sequences = filter_sequences(
                pos_n_sequences, includes_pos, excludes_pos
            )
            pos_sequence_counts = pd.Series(filtered_sequences).value_counts()
            sorted_pos_sequence_counts[lang] = {
                k: (v.item() if hasattr(v, "item") else v)
                for k, v in pos_sequence_counts.items()
            }
        return sorted_pos_sequence_counts

    @time_it
    def generate_word_dict(
        self, pos_word_counts: Dict[str, Dict[str, int]], n: int
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        results_dict: Dict[str, Dict[str, Dict[str, int]]] = {}
        for lang, pos_word_counts_lang in tqdm(
            pos_word_counts.items(), desc="Generating word dicts"
        ):
            word_dict: Dict[str, Dict[str, int]] = {}
            for wordpos, word_counts in pos_word_counts_lang.items():
                words_and_pos = wordpos.split()
                words = [w.split("/")[0] for w in words_and_pos if "/" in w]
                pos_tags = [w.split("/")[1] for w in words_and_pos if "/" in w]
                word_sequence = " ".join(words)
                pos_sequence = " ".join(pos_tags)
                if len(pos_sequence.strip().split(" ")) != n:
                    continue
                if not pos_sequence or not word_sequence:
                    continue
                if pos_sequence not in word_dict:
                    word_dict[pos_sequence] = {}
                word_dict[pos_sequence][word_sequence] = word_counts
            results_dict[lang] = word_dict
        return results_dict

    def generate_word_texts(
        self,
        word_dict: Dict[str, Dict[str, Dict[str, int]]],
        languages: List[str],
        pos_text_count: Optional[int] = None,
    ) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        all_pos_texts_dict = {
            d["pos_text"]: {"lang": d["lang"], "text": d["text"]} for d in self.data
        }
        word_texts: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
        for lang in languages:
            allowed_pos_sequences_per_tag: Dict[str, List[str]] = defaultdict(list)
            for pos_tag, word_dict_lang in word_dict[lang].items():
                pos_tag = pos_tag.strip()
                if pos_tag in ["PART", "PUNCT"]:
                    continue
                words = list(word_dict_lang.keys())
                for word in words:
                    if pos_tag not in allowed_pos_sequences_per_tag:
                        allowed_pos_sequences_per_tag[pos_tag] = []
                    allowed_pos_sequences_per_tag[pos_tag].append(word)
            allowed_pos_sequences = []
            for pos_sequences in allowed_pos_sequences_per_tag.values():
                allowed_pos_sequences.extend(pos_sequences[:pos_text_count])
            for pos_sequence in tqdm(
                allowed_pos_sequences, desc=f"Generating '{lang}' word texts"
            ):
                for pos_text, pos_dict in all_pos_texts_dict.items():
                    text = pos_dict["text"]
                    lang = pos_dict["lang"]
                    if lang not in word_texts:
                        word_texts[lang] = {}
                    all_texts = word_texts[lang].values()
                    pattern = r"\b" + re.escape(pos_sequence) + r"\b"
                    if pos_sequence not in word_texts[lang] and re.search(
                        pattern, pos_text
                    ):
                        capitalized_text = text.capitalize()
                        if capitalized_text in all_texts:
                            continue
                        if pos_sequence not in word_texts[lang]:
                            word_texts[lang][pos_sequence] = []
                        elif (
                            pos_text_count
                            and len(word_texts[lang][pos_sequence]) >= pos_text_count
                        ):
                            continue
                        word_texts[lang][pos_sequence].append(capitalized_text)
                        break
        formatted_pos_texts: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
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
                sorted_words = dict(sorted(formatted_pos_texts[lang][pos].items()))
                formatted_pos_texts[lang][pos] = sorted_words
            word_texts[lang] = formatted_pos_texts[lang]
        return word_texts

    @time_it
    def extract_data(
        self,
        n: int = 1,
        min_n: int = 1,
        max_n: Optional[int] = None,
        includes_pos: Optional[List[str]] = None,
        excludes_pos: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        def generate_pos_text_sequences(text_list: List[str], n: int) -> List[str]:
            return [
                " ".join([text for text in text_seq])
                for text_seq in self.nwise(text_list, n)
            ]

        def filter_sequences(
            sequences: List[str],
            includes: Optional[List[str]],
            excludes: Optional[List[str]],
            min_n: int,
            max_n: Optional[int],
        ) -> Dict[str, Any]:
            matched_word_count: Dict[str, Dict[str, int]] = {}
            global_word_count: Dict[str, Dict[str, int]] = {}
            for seq in sequences:
                for pos in includes or []:
                    if pos not in matched_word_count:
                        matched_word_count[pos] = {}
                        global_word_count[pos] = {}
                    matched_pos_words = re.findall(rf"\b([\w,:-]+)/{pos}\b", seq)
                    for word in matched_pos_words:
                        if word not in matched_word_count[pos]:
                            matched_word_count[pos][word] = 0
                            global_word_count[pos][word] = 0
                        global_word_count[pos][word] += 1
            global_filtered_sequences = []
            for seq in sequences:
                include_seq = True
                for pos in includes or []:
                    matched_pos_words = re.findall(rf"\b([\w,:-]+)/{pos}\b", seq)
                    if not matched_pos_words or any(
                        global_word_count[pos][word] < min_n
                        for word in matched_pos_words
                    ):
                        include_seq = False
                        break
                if include_seq:
                    global_filtered_sequences.append(seq)
            filtered_sequences = []
            removed_sequence_words = []
            for seq in global_filtered_sequences:
                include_seq = True
                if (not includes or any(f"/{pos}" in seq for pos in includes)) and (
                    not excludes or not any(f"/{pos}" in seq for pos in excludes)
                ):
                    word_pos_pairs = re.findall(r"\b([\w,:-]+)/([\w,:-]+)\b", seq)
                    for item in word_pos_pairs:
                        word = item[0]
                        pos = item[1]
                        if pos in (includes or []):
                            count = matched_word_count[pos][word]
                            if max_n and count > max_n:
                                include_seq = False
                if include_seq:
                    word_pos_pairs = re.findall(r"\b([\w,:-]+)/([\w,:-]+)\b", seq)
                    for item in word_pos_pairs:
                        word = item[0]
                        pos = item[1]
                        if pos in (includes or []):
                            matched_word_count[pos][word] += 1
                    filtered_sequences.append(seq)
            sorted_word_count_in_desc: Dict[str, Dict[str, int]] = {}
            for pos, words in matched_word_count.items():
                sorted_word_count_in_desc[pos] = dict(
                    sorted(words.items(), key=lambda item: item[1], reverse=True)
                )
            removed_sequence_words_dict = {
                pos: [word for word, count in words.items() if count == 0]
                for pos, words in sorted_word_count_in_desc.items()
                if pos in (includes or [])
            }
            sorted_word_count_in_desc = {
                pos: {word: count for word, count in words.items() if count > 0}
                for pos, words in sorted_word_count_in_desc.items()
            }
            return {
                "word_count": sorted_word_count_in_desc,
                "sequences": filtered_sequences,
                "removed_sequence_words": removed_sequence_words_dict,
            }

        result_dict: Dict[str, Any] = {}
        word_count: Dict[str, Dict[str, int]] = {}
        for lang, data in tqdm(
            self.grouped_data.items(), desc="Generating POS word counts"
        ):
            pos_text_n_sequences = generate_pos_text_sequences(data["texts"], n)
            result_dict[lang] = filter_sequences(
                pos_text_n_sequences, includes_pos, excludes_pos, min_n, max_n
            )
            word_count = result_dict[lang]["word_count"]
            removed_sequence_words = result_dict[lang]["removed_sequence_words"]
        filtered_data = []
        pos_text_dict = {d["pos_text"]: d for d in self.data}
        for lang, lang_data in result_dict.items():
            for pos_text in lang_data["sequences"]:
                if pos_text in pos_text_dict:
                    filtered_data.append(pos_text_dict[pos_text])
        filtered_sequences = set(d["pos_text"] for d in filtered_data)
        remaining_data = [
            d for d in self.data if d["pos_text"] not in filtered_sequences
        ]
        return {
            "data": filtered_data,
            "remaining_data": remaining_data,
            "info": {
                "word_count": word_count,
                "removed_sequence_words": removed_sequence_words,
            },
        }

    def generate_analysis(
        self,
        n: int = 1,
        top_n: int = 10,
        all_combinations: bool = False,
        includes_pos: List[str] = None,
        excludes_pos: List[str] = None,
        words_only: bool = False,
        word_texts: bool = False,
        pos_text_count: Optional[int] = None,
    ):
        languages = self.languages
        pos_word_counts = self.get_pos_word_counts(n, includes_pos, excludes_pos)
        yield {
            "type": "pos_word_counts",
            "languages": languages,
            "data": pos_word_counts,
        }
        if words_only or word_texts:
            word_dict = self.generate_word_dict(pos_word_counts, n)
            yield {"type": "word_dict", "languages": languages, "data": word_dict}
            if word_texts:
                word_texts_result = self.generate_word_texts(
                    word_dict, languages, pos_text_count
                )
                yield {
                    "type": "word_texts",
                    "languages": languages,
                    "data": word_texts_result,
                }
        else:
            pos_sequence_counts = self.get_pos_sequence_counts(
                n, includes_pos, excludes_pos
            )
            yield {
                "type": "pos_sequence_counts",
                "languages": languages,
                "data": pos_sequence_counts,
            }
        language_results = self.process_language_data(
            n, top_n, all_combinations, includes_pos, excludes_pos
        )
        yield {
            "type": "language_results",
            "languages": languages,
            "data": language_results,
        }


def clean_text(text: str) -> str:
    """Clean a text by removing [EOS], [BOS], and their variations with slashes."""
    cleaned_text = re.sub(r"\[EOS\]|\[BOS\]", "", text)
    cleaned_text = re.sub(r"/?\[EOS\]/?|/?\[BOS\]/?", "", cleaned_text)
    return cleaned_text.strip(" /")


def clean_dictionary(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively clean all keys and values in a dictionary, ensuring keys that are meaningful after cleaning are retained."""
    cleaned_dict: Dict[str, Any] = {}
    for key, value in data.items():
        cleaned_key = clean_text(key).strip()
        if cleaned_key or key.strip():
            if isinstance(value, dict):
                cleaned_value = clean_dictionary(value)
            elif isinstance(value, str):
                cleaned_value = clean_text(value)
            else:
                cleaned_value = value
            cleaned_dict[cleaned_key if cleaned_key else key.strip()] = cleaned_value
    return cleaned_dict


def _remove_redundant_subsets(counts_dict: Dict[str, int]) -> Dict[str, int]:
    """
    Remove subset ngrams that have the same count as their superset.
    For example, if "Learning Engineer" (2-gram) and "Machine Learning Engineer" (3-gram)
    both have count=24, remove "Learning Engineer" since it's a subset of the longer
    ngram with the same frequency.

    Args:
        counts_dict: Dictionary mapping ngram strings to their counts

    Returns:
        Filtered dictionary with redundant subsets removed
    """
    sorted_items = sorted(
        counts_dict.items(), key=lambda x: (-len(x[0].split()), -x[1])
    )

    kept_ngrams: List[Tuple[str, int]] = []
    filtered_dict: Dict[str, int] = {}

    for ngram, count in sorted_items:
        is_subset = False
        for kept_ngram, kept_count in kept_ngrams:
            if kept_count == count and ngram in kept_ngram:
                is_subset = True
                break

        if not is_subset:
            filtered_dict[ngram] = count
            kept_ngrams.append((ngram, count))

    return filtered_dict


def _deep_merge_counts(
    target: Dict[str, Any], source: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries, adding numeric values and merging nested dicts.

    Args:
        target: The target dictionary to merge into
        source: The source dictionary to merge from

    Returns:
        Merged dictionary
    """
    for key, value in source.items():
        if key in target:
            if isinstance(value, dict) and isinstance(target[key], dict):
                # Recursively merge nested dictionaries
                target[key] = _deep_merge_counts(target[key], value)
            elif isinstance(value, (int, float)) and isinstance(
                target[key], (int, float)
            ):
                # Add numeric values
                target[key] += value
            else:
                # For non-matching types, keep the source value (or could raise an error)
                target[key] = value
        else:
            target[key] = value
    return target


def analyze_pos_tags(
    texts: List[Dict[str, str]],
    n: Optional[int] = None,
    top_n: int = 10,
    source_key: str = "text",
    words_only: bool = False,
    word_texts: bool = False,
    all_combinations: bool = False,
    includes_pos: Optional[List[str]] = None,
    excludes_pos: Optional[List[str]] = None,
    pos_text_count: Optional[int] = None,
    ngram_range: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    Analyze POS tags in texts.

    Args:
        texts: List of text dictionaries with 'text' and optionally 'lang' keys
        n: N-gram size (optional, defaults based on ngram_range)
        top_n: Number of top results to return
        source_key: Key to use for text in input dictionaries
        words_only: If True, return word dictionaries instead of POS sequences
        word_texts: If True, also generate word texts
        all_combinations: If True, use all combinations instead of sliding window
        includes_pos: List of POS tags to include
        excludes_pos: List of POS tags to exclude
        pos_text_count: Maximum number of texts per POS sequence
        ngram_range: Optional tuple (min_n, max_n) to analyze multiple n-gram sizes

    Returns:
        Dictionary containing analysis results with descriptive keys. If ngram_range is provided,
        results are merged across all n-gram sizes with redundant subsets removed.
    """
    console.print("\n[bold blue]=== Starting POS Tag Analysis ===[/bold blue]")
    console.print(f"[cyan]Number of texts:[/cyan] {len(texts)}")
    console.print(f"[cyan]ngram_range:[/cyan] {ngram_range}")
    console.print(f"[cyan]words_only:[/cyan] {words_only}")
    console.print(f"[cyan]includes_pos:[/cyan] {includes_pos}")
    console.print(f"[cyan]excludes_pos:[/cyan] {excludes_pos}")

    tagger = POSTagger()
    tagged_data: List[Dict[str, Any]] = []

    console.print("\n[yellow]Tagging texts...[/yellow]")
    for idx, text_dict in enumerate(texts):
        text = text_dict[source_key]
        lang = text_dict.get("lang", "en")
        pos_results = tagger.process_and_tag(text)
        tagged_text = tagger.format_tags(pos_results)
        tagged_data.append(
            {"lang": lang, "text": text, "pos": pos_results, "pos_text": tagged_text}
        )
        if (idx + 1) % 10 == 0 or idx == len(texts) - 1:
            console.print(f"  Tagged {idx + 1}/{len(texts)} texts")

    console.print(f"[green]✓ Tagged {len(tagged_data)} texts[/green]")

    includes_pos = includes_pos or []
    excludes_pos = excludes_pos or []

    if ngram_range is not None:
        min_n, max_n = ngram_range
        console.print(
            f"\n[yellow]Analyzing n-gram range: {min_n} to {max_n} (optimized single-pass)[/yellow]"
        )

        console.print(
            f"\n[bold cyan]--- Processing all n-grams in single pass ---[/bold cyan]"
        )

        # Create processor with max_n to get all POS data
        processor = LanguageDataProcessor(
            tagged_data,
            n=max_n,
            format_lambda=text_lambda,
        )

        # Initialize merged results
        merged_results: Dict[str, Any] = {}

        # Process pos_word_counts for all n-gram sizes
        console.print(
            "\n  [dim]Generating POS word counts for all n-gram sizes...[/dim]"
        )
        for current_n in range(min_n, max_n + 1):
            pos_word_counts = processor.get_pos_word_counts(
                current_n, includes_pos, excludes_pos
            )

            key = "pos_word_counts_all"
            if key not in merged_results:
                merged_results[key] = {}

            for lang, lang_counts in pos_word_counts.items():
                if lang not in merged_results[key]:
                    merged_results[key][lang] = {}
                _deep_merge_counts(merged_results[key][lang], lang_counts)

        console.print(f"    [green]✓ Merged pos_word_counts[/green]")

        # Process word_dict for all n-gram sizes
        if words_only or word_texts:
            console.print(
                "\n  [dim]Generating word dicts for all n-gram sizes...[/dim]"
            )
            for current_n in range(min_n, max_n + 1):
                # Need to regenerate pos_word_counts for this n
                pos_word_counts = processor.get_pos_word_counts(
                    current_n, includes_pos, excludes_pos
                )
                word_dict = processor.generate_word_dict(pos_word_counts, current_n)

                key = "word_dict_all"
                if key not in merged_results:
                    merged_results[key] = {}

                for lang, lang_dict in word_dict.items():
                    if lang not in merged_results[key]:
                        merged_results[key][lang] = {}
                    _deep_merge_counts(merged_results[key][lang], lang_dict)

            console.print(f"    [green]✓ Merged word_dict[/green]")

            # Process word_texts if requested
            if word_texts:
                console.print("\n  [dim]Generating word texts...[/dim]")
                # Note: word_texts generation is complex and may need adjustment for multi-n merge
                console.print(
                    f"    [yellow]⚠ Word texts generation skipped (complex multi-n merge)[/yellow]"
                )

        # Process pos_sequence_counts for all n-gram sizes
        if not (words_only or word_texts):
            console.print(
                "\n  [dim]Generating POS sequence counts for all n-gram sizes...[/dim]"
            )
            for current_n in range(min_n, max_n + 1):
                pos_sequence_counts = processor.get_pos_sequence_counts(
                    current_n, includes_pos, excludes_pos
                )

                key = "pos_sequence_counts_all"
                if key not in merged_results:
                    merged_results[key] = {}

                for lang, lang_counts in pos_sequence_counts.items():
                    if lang not in merged_results[key]:
                        merged_results[key][lang] = {}
                    _deep_merge_counts(merged_results[key][lang], lang_counts)

            console.print(f"    [green]✓ Merged pos_sequence_counts[/green]")

        # Process language_results for all n-gram sizes
        console.print(
            "\n  [dim]Generating language results for all n-gram sizes...[/dim]"
        )
        for current_n in range(min_n, max_n + 1):
            language_results = processor.process_language_data(
                current_n, top_n, all_combinations, includes_pos, excludes_pos
            )

            key = "language_results_all"
            if key not in merged_results:
                merged_results[key] = {}

            for lang, lang_results in language_results.items():
                if lang not in merged_results[key]:
                    merged_results[key][lang] = {}

                for metric, value in lang_results.items():
                    if metric in [
                        "top_most_common_ngrams",
                        "top_least_common_ngrams",
                        "top_most_common_pos_sequences",
                        "top_least_common_pos_sequences",
                    ]:
                        if metric not in merged_results[key][lang]:
                            merged_results[key][lang][metric] = {}
                        _deep_merge_counts(merged_results[key][lang][metric], value)
                    else:
                        # For scalar metrics, keep the last value
                        merged_results[key][lang][metric] = value

        console.print(f"    [green]✓ Merged language_results[/green]")

        # Apply subset removal to count-based results
        console.print(f"\n[yellow]Removing redundant subsets...[/yellow]")

        for key in merged_results:
            data = merged_results[key]
            if isinstance(data, dict):
                # Check if it's flat count data
                if all(isinstance(v, (int, float)) for v in data.values()):
                    original_count = len(data)
                    merged_results[key] = _remove_redundant_subsets(data)
                    removed = original_count - len(merged_results[key])
                    if removed > 0:
                        console.print(
                            f"  [yellow]{key}: Removed {removed} redundant subsets[/yellow]"
                        )
                # Check if it's nested dict with count values
                elif all(isinstance(v, dict) for v in data.values()):
                    cleaned = {}
                    total_removed = 0
                    for outer_key, inner_dict in data.items():
                        if inner_dict and all(
                            isinstance(v, (int, float)) for v in inner_dict.values()
                        ):
                            original_count = len(inner_dict)
                            cleaned[outer_key] = _remove_redundant_subsets(inner_dict)
                            total_removed += original_count - len(cleaned[outer_key])
                        else:
                            cleaned[outer_key] = inner_dict
                    if total_removed > 0:
                        console.print(
                            f"  [yellow]{key}: Removed {total_removed} redundant subsets from nested dicts[/yellow]"
                        )
                    merged_results[key] = cleaned

        console.print(f"\n[bold green]=== Analysis Complete ===[/bold green]")
        console.print(f"[green]Total result keys: {len(merged_results)}[/green]")
        for key in sorted(merged_results.keys()):
            val = merged_results[key]
            if isinstance(val, dict):
                console.print(f"  [dim]{key}: {len(val)} items[/dim]")
            else:
                console.print(f"  [dim]{key}: {type(val).__name__}[/dim]")

        return merged_results

    else:
        # Use single n value (default to 1 if not provided)
        if n is None:
            n = 1

        console.print(f"\n[yellow]Processing single n={n}[/yellow]")

        processor = LanguageDataProcessor(tagged_data, n=n, format_lambda=text_lambda)
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

        results: Dict[str, Any] = {}

        for result in result_stream:
            result_type = result["type"]
            languages = result["languages"]
            data = result["data"]

            console.print(f"  [dim]Processing: {result_type}[/dim]")

            for lang in languages:
                cleaned_result = clean_dictionary(data[lang])
                if result_type == "word_dict":
                    results["word_dict_all"] = cleaned_result
                elif result_type == "word_texts":
                    results["word_texts_all"] = cleaned_result
                elif result_type == "pos_word_counts":
                    results["pos_word_counts_all"] = cleaned_result
                elif result_type == "pos_sequence_counts":
                    results["pos_sequence_counts_all"] = cleaned_result
                elif result_type == "language_results":
                    results["language_results_all"] = cleaned_result

        console.print(f"\n[bold green]=== Analysis Complete ===[/bold green]")
        console.print(f"[green]Total result keys: {len(results)}[/green]")

        return results
