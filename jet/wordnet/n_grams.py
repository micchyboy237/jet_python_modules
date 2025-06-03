from jet.wordnet.pos_tagger import POSItem, POSTagEnum, POSTagType, POSTagger
from itertools import tee, islice
from typing import List
from jet.wordnet.sentence import split_by_punctuations, split_sentences
from nltk import word_tokenize, ngrams
import string
from collections import Counter
from typing import Optional, List, Tuple, Union, Dict
from collections import defaultdict, Counter
from collections import defaultdict
import itertools
from jet.wordnet.words import count_words
from jet.wordnet.words import get_words
from functools import lru_cache
from tqdm import tqdm
from jet.wordnet.stopwords import StopWords
from jet.wordnet.similarity import filter_different_texts


def get_ngrams(texts: Union[str, List[str]], min_words: int = 1, min_count: Optional[int] = None, max_words: Optional[int] = None) -> List[str]:
    ngrams_dict = count_ngrams(
        texts, min_words=min_words, min_count=min_count, max_words=max_words)
    return list(ngrams_dict.keys())


def separate_ngram_lines(texts: str | list[str], punctuations_split: list[str] = [',', '/', ':']) -> list[str]:
    if isinstance(texts, str):
        texts = [texts]
    results: list[str] = []
    for text in texts:
        # Remove empty strings and strip whitespace
        text = text.strip()
        if not text:
            continue
        sentences = split_sentences(text)
        sentences = list(set(sentences))  # Remove duplicates
        for sentence in sentences:
            sub_sentences = split_by_punctuations(
                sentence.strip(), punctuations_split)
            results.extend([s.strip() for s in sub_sentences if s.strip()])
    # Sort results to ensure consistent order
    return sorted(results)


def extract_ngrams(texts: Union[str, List[str]], min_words: int = 1, max_words: int = 1) -> list[str]:
    if isinstance(texts, str):
        texts = [texts]
    texts = separate_ngram_lines(texts)
    ngrams: list[str] = []
    for text in texts:
        n_min = min_words
        n_max = max_words
        if n_max > n_min:
            for n_val in range(n_min, n_max + 1):
                ngrams.extend(get_words(text, n_val))
        else:
            ngrams.extend(get_words(text, n_min))
    return ngrams


def count_ngrams(texts: Union[str, List[str]], min_words: int = 1, min_count: Optional[int] = None, max_words: Optional[int] = None, from_start: bool = False):
    if isinstance(texts, str):
        texts = [texts]
    # Handle empty input
    if not texts or all(not text.strip() for text in texts):
        return {}
    if not max_words:
        max_words = max((count_words(text)
                        for text in texts if text.strip()), default=1)
    if from_start:
        sentences = separate_ngram_lines(texts)
        ngrams = []
        for sentence in sentences:
            words = sentence.split()
            for n in range(min_words, min(max_words, len(words)) + 1):
                ngrams.append(" ".join(words[:n]))
    else:
        ngrams = extract_ngrams(
            texts, min_words=min_words, max_words=max_words)
    ngram_counter = Counter(ngrams)
    ngrams_dict = {ngram: count for ngram, count in ngram_counter.items(
    ) if not min_count or count >= min_count}
    return ngrams_dict


def get_most_common_ngrams(texts: Union[str, List[str]], min_words: int = 1, min_count: int = 2, max_words: Optional[int] = None) -> dict[str, int]:
    if isinstance(texts, str):
        texts = [texts]
    texts = [text.lower() for text in texts if text.strip()]
    if not texts:
        return {}
    ngrams_dict = count_ngrams(texts, min_words=min_words,
                               min_count=min_count, max_words=max_words)
    stopwords = StopWords()
    filtered_ngrams = {}
    for (ngram, count) in ngrams_dict.items():
        ngram_words = get_words(ngram)
        start = ngram_words[0]
        end = ngram_words[-1]
        if start in stopwords.english_stop_words or end in stopwords.english_stop_words:
            continue
        filtered_ngrams[ngram] = count
    return filtered_ngrams


def count_ngrams_with_texts(
    texts: List[str],
    min_words: int = 1,
    min_count: int = 2,
    max_words: Optional[int] = None
) -> List[dict]:
    ngrams_dict = {}
    for text_idx, text in enumerate(texts):
        words = get_words(text)
        for n in range(min_words, max_words or len(words) + 1):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i + n])
                if ngram not in ngrams_dict:
                    ngrams_dict[ngram] = {'count': 0, 'texts': set()}
                ngrams_dict[ngram]['count'] += 1
                ngrams_dict[ngram]['texts'].add(text_idx)
    result = [
        {
            'ngram': ngram,
            'count': data['count'],
            'texts': list(data['texts'])
        }
        for ngram, data in ngrams_dict.items()
        if data['count'] >= min_count
    ]
    return result


def group_sentences_by_ngram(
    sentences: List[str],
    min_words: int = 2,
    top_n: int = 2,
    is_start_ngrams: bool = True,
    includes_pos: List[POSTagType] = []
) -> Dict[str, List[str]]:
    sentence_ngrams = defaultdict(list)
    tagger = POSTagger()
    includes_pos_lower = [pos.value.lower() if isinstance(
        pos, POSTagEnum) else pos.lower() for pos in includes_pos]
    for sentence in tqdm(sentences, desc="Grouping sentences"):
        ngrams_list = get_words(sentence, min_words)
        if not ngrams_list:
            continue
        valid_ngrams = []
        if includes_pos:
            for ngram in ngrams_list:
                pos_results = tagger.process_and_tag(ngram)
                all_words_valid = all(
                    any(
                        pos.lower() in includes_pos_lower
                        for pos in (pos_result['pos'] if isinstance(pos_result['pos'], list) else [pos_result['pos']])
                    )
                    for pos_result in pos_results
                )
                if all_words_valid:
                    valid_ngrams.append(ngram)
        else:
            valid_ngrams = ngrams_list
        if is_start_ngrams and valid_ngrams:
            sentence_ngrams[valid_ngrams[0]].append(sentence)
        elif not is_start_ngrams:
            unique_ngrams = set(valid_ngrams)
            for ngram in unique_ngrams:
                sentence_ngrams[ngram].append(sentence)
    optimized_groups = {}
    for ngram, group_sentences in tqdm(sentence_ngrams.items(), desc="Optimizing groups"):
        sorted_group_sentences = sorted(
            group_sentences, key=lambda s: (count_words(s), sentences.index(s)))
        optimized_groups[ngram] = sorted_group_sentences[:top_n]
    return optimized_groups


def n_gram_frequency(sentence, n=2):
    """ Calculate the frequency of n-grams in a sentence """
    n_grams = [sentence[i:i+n] for i in range(len(sentence) - n + 1)]
    return Counter(n_grams)


def calculate_n_gram_diversity(freq):
    """ Calculate diversity based on the count of unique n-grams """
    return len(freq)


def get_ngram_weight(all_ngrams, sentence_ngrams, previous_ngrams):
    penalty = sum(ngram in previous_ngrams for ngram in sentence_ngrams)
    return sum(1 / all_ngrams[ngram] for ngram in sentence_ngrams if ngram in all_ngrams) + penalty


def sort_sentences(sentences, n):
    all_ngrams = Counter()
    sentence_ngrams_dict = {}
    for sentence in tqdm(sentences, desc="Precomputing n-grams"):
        ngram_list = get_words(sentence, n)
        all_ngrams.update(ngram_list)
        sentence_ngrams_dict[sentence] = ngram_list
    sorted_sentences = sorted(sentences)
    return sorted_sentences


def filter_and_sort_sentences_by_ngrams(sentences: List[str], min_words: int = 2, top_n: int = 2, is_start_ngrams=True) -> List[str]:
    sentence_ngrams = defaultdict(list)
    all_ngrams = Counter()
    for sentence in tqdm(sentences, desc="Grouping sentences"):
        ngrams_list = get_words(sentence, min_words)
        all_ngrams.update(ngrams_list)
        if is_start_ngrams and ngrams_list:
            sentence_ngrams[ngrams_list[0]].append(sentence)
        elif not is_start_ngrams:
            for ngram in set(ngrams_list):
                sentence_ngrams[ngram].append(sentence)
    most_common_ngrams = [ngram for ngram, _ in all_ngrams.most_common(top_n)]
    filtered_sentences = set()
    for ngram, group_sentences in sentence_ngrams.items():
        if ngram in most_common_ngrams:
            filtered_sentences.update(group_sentences[:top_n])
    sorted_sentences = sorted(filtered_sentences)
    return sorted_sentences


def filter_and_sort_sentences_by_similarity(sentences: List[str], min_words=2, threshold=0.8) -> List[str]:
    filtered_sentences = filter_different_texts(sentences, threshold)
    sorted_sentences = sort_sentences(filtered_sentences, min_words)
    return sorted_sentences


def recursive_filter(texts: List[str], max_n: int, depth: int = 0, max_depth: int = 10) -> Tuple[List[str], List[str]]:
    if depth > max_depth:
        return texts, []
    ngram_counter = Counter()
    text_ngram_counts = {}
    for text in texts:
        text_ngrams = count_ngrams(text, max_n)
        text_ngram_counts[text] = text_ngrams
        ngram_counter.update(text_ngrams)
    passed_texts = []
    failed_texts = []
    for text, ngrams in text_ngram_counts.items():
        if all(ngram_counter[ngram] <= max_n for ngram in ngrams):
            passed_texts.append(text)
        else:
            failed_texts.append(text)
    if failed_texts:
        additional_passed, _ = recursive_filter(
            failed_texts, max_n, depth + 1, max_depth)
        passed_texts.extend(additional_passed)
    return passed_texts, failed_texts


def get_total_unique_ngrams(ngram_counter):
    return len(ngram_counter)


def get_total_counts_of_ngrams(ngram_counter):
    return sum(ngram_counter.values())


def get_specific_ngram_count(ngram_counter, specific_ngram):
    return ngram_counter[specific_ngram]


def get_ngrams_by_range(
    texts: Union[str, List[str]],
    min_words: int,
    count: Optional[Union[int, Tuple[int, int]]] = None,
    max_words: int = 1,
    show_count: bool = False
) -> Union[List[str], List[Dict[str, int]]]:
    if isinstance(texts, str):
        texts = [texts]
    ngram_counter = Counter()
    for text in texts:
        ngram_counter.update(count_ngrams(
            text, min_words=min_words, max_words=max_words))
    results = []

    def add_result(ngram, count):
        if show_count:
            results.append({"ngram": ngram, "count": count})
        else:
            results.append(ngram)
    for ngram, n_count in ngram_counter.items():
        if isinstance(count, tuple):
            count_min, count_max = (count[0], float(
                'inf')) if len(count) == 1 else count
            if count_min <= n_count <= count_max:
                add_result(ngram, n_count)
        elif isinstance(count, int) and count <= n_count:
            add_result(ngram, n_count)
        elif count is None:
            add_result(ngram, n_count)
    return results


def filter_texts_by_multi_ngram_count(
        texts: List[str],
        min_words: int,
        count: Union[int, Tuple[int, int]],
        max_words: int = 1,
        count_all_ngrams: bool = True,
) -> List[str]:
    ngrams_list = get_ngrams_by_range(
        texts, min_words=min_words, count=count, max_words=max_words, show_count=True)
    ngrams_dict = {ngram_dict['ngram']: ngram_dict['count']
                   for ngram_dict in ngrams_list}
    if isinstance(count, tuple):
        count_min, count_max = (count[0], float(
            'inf')) if len(count) == 1 else count
    else:
        count_min = count
        count_max = float('inf')
    filtered_texts = []
    if count_all_ngrams:
        for text in tqdm(texts, desc="Filtering texts"):
            text_words = get_words(text, min_words)
            if all(word in ngrams_dict for word in text_words):
                filtered_texts.append(text)
    else:
        for text in tqdm(texts, desc="Filtering texts"):
            match_ngrams_dict = {
                ngram: ngrams_dict[ngram] for ngram in ngrams_dict if ngram in text}
            n_counts = match_ngrams_dict.values()
            has_violating_count = False
            for n_count in n_counts:
                if not count_min <= n_count <= count_max:
                    has_violating_count = True
                    break
            if has_violating_count or not match_ngrams_dict:
                continue
            filtered_texts.append(text)
    return filtered_texts


def nwise(iterable, n=1):
    "Returns a sliding window (of width n) over data from the iterable"
    iters = tee(iterable, n)
    for i, it in enumerate(iters):
        next(islice(it, i, i), None)
    return zip(*iters)


def get_common_texts(texts, includes_pos=["PROPN", "NOUN", "VERB", "ADJ"], min_words: int = 1, max_words: Optional[int] = None):
    if not texts:
        return []
    tagger = POSTagger()
    stopwords = StopWords()
    # Process each text to get filtered words based on POS and stopwords
    filtered_texts = []
    text_ngrams_list = []
    for text in tqdm(texts, desc="Processing texts"):
        text = text.lower().strip()
        if not text:
            continue
        pos_results = tagger.filter_pos(text, includes_pos)
        filtered_words = [pos_result['word'] for pos_result in pos_results if pos_result['word'].lower(
        ) not in stopwords.english_stop_words]
        filtered_text = " ".join(filtered_words)
        if filtered_text:
            filtered_texts.append(filtered_text)
            # Generate n-grams of max_words length for this text
            ngrams = get_words(filtered_text, max_words or min_words)
            text_ngrams_list.append(set(ngrams))
    if not filtered_texts:
        return []
    # Find n-grams common to all texts
    if text_ngrams_list:
        common_ngrams = set.intersection(*text_ngrams_list)
    else:
        common_ngrams = set()
    # Filter n-grams to ensure they have exactly max_words words
    if max_words:
        common_ngrams = {
            ngram for ngram in common_ngrams if count_words(ngram) == max_words}
    return sorted(list(common_ngrams))


if __name__ == "__main__":
    texts = [
        "Describe the structure of an important roadmap.",
        "Give three tips",
        "How can we reduce this?",
        "You need to make a tough decision for an important roadmap.",
        "Identify the odd one out.",
        "Explain why",
        "Write a short story",
        "Describe the structure of hair",
        "Because the structure is important",
    ]
    print("\nAll n-grams:")
    all_ngrams = count_ngrams([text.lower() for text in texts], min_words=1)
    for ngram, count in all_ngrams.items():
        print(f"{ngram}: {count}")
    print(f"\nFiltered n-grams by min_count:")
    filtered_ngrams = count_ngrams([text.lower()
                                   for text in texts], min_count=2)
    for ngram, count in filtered_ngrams.items():
        print(f"{ngram}: {count}")
    print("\nMost Common n-grams:")
    result = get_most_common_ngrams([text.lower() for text in texts])
    print(result)
    print("\nCommon texts:")
    result = get_common_texts([text.lower()
                              for text in texts], includes_pos=["PROPN", "NOUN"])
    print(result)
    print("\nGrouped sentences by ngram:")
    result = group_sentences_by_ngram([text.lower()
                                       for text in texts], is_start_ngrams=False)
    print(result)
    specific_ngrams = count_ngrams([text.lower()
                                   for text in texts], min_words=1, max_words=3)
    print(
        f"\nTotal unique n-grams: {get_total_unique_ngrams(specific_ngrams)}")
    print(
        f"\nTotal counts of all n-grams: {get_total_counts_of_ngrams(specific_ngrams)}")
    print("\nN-grams of Specific Range:")
    for ngram_dict in get_ngrams_by_range(texts, min_words=1, max_words=2, count=(2, ), show_count=True):
        print(f"{ngram_dict['ngram']}: {ngram_dict['count']}")
    print("\nN-grams of Specific Count:")
    for ngram in get_ngrams_by_range(texts, min_words=2, count=2, show_count=True):
        print(f"{ngram}")
    results = filter_texts_by_multi_ngram_count(
        texts, min_words=1, count=(2, ), count_all_ngrams=True)
    print(
        f"\nFilter texts by multi-ngram count: {len(results)}\nOriginal: {len(texts)}")
    print(results)
