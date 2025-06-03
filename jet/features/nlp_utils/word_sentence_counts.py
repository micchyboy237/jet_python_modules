from typing import Tuple, Union, List, Dict, Optional, TypedDict
import itertools
from jet.wordnet.sentence import split_sentences
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
from functools import partial

from jet.features.nlp_utils.nlp_types import NgramOccurrence, Matched
from jet.features.nlp_utils.utils import LEMMATIZER, STOP_WORDS, get_wordnet_pos


def process_single_text(single_text: str, n: Optional[int] = None, min_count: int = 1,
                        in_sequence: bool = False, max_n: int = 10) -> List[Matched]:
    occurrences: Dict[int, Dict[Tuple[str, ...], NgramOccurrence]] = defaultdict(
        lambda: defaultdict(lambda: {
            'count': 0,
            'start_idxs': [],
            'end_idxs': [],
            'sentence_idx': 0,
            'sentence': '',
            'ngram': ()
        })
    )
    sentences = split_sentences(single_text)
    sentence_words: List[Tuple[int, List[Tuple[str, int, int]], str]] = []
    char_offset = 0
    if not n or n > 1:
        in_sequence = True
    for sentence_idx, sentence in enumerate(sentences):
        tokens = word_tokenize(sentence.lower())
        words = [(token, i) for i, token in enumerate(tokens)
                 if token.isalpha() and token not in STOP_WORDS]
        tagged_words = pos_tag([w[0] for w in words])
        lemmatized_words = [
            (LEMMATIZER.lemmatize(word, get_wordnet_pos(tag)), idx)
            for (word, idx), (word_tagged, tag) in zip(words, tagged_words)
        ]
        word_positions = []
        current_pos = char_offset
        for word, _ in words:
            start_idx = single_text[current_pos:].lower().find(
                word) + current_pos
            end_idx = start_idx + len(word)
            word_positions.append((word, start_idx, end_idx))
            current_pos = end_idx
        lemmatized_with_positions = [
            (lem_word, start_idx, end_idx)
            for (lem_word, _), (_, start_idx, end_idx) in zip(lemmatized_words, word_positions)
        ]
        if lemmatized_with_positions:
            sentence_words.append(
                (sentence_idx, lemmatized_with_positions, sentence))
        char_offset += len(sentence) + 1
    if n is None:
        max_n_local = min(max(len(words) for _, words,
                          _ in sentence_words) if sentence_words else 1, max_n)
        combination_sizes = range(1, max_n_local + 1)
    else:
        combination_sizes = [min(n, max_n)]
    for sentence_idx, lemmatized_with_positions, sentence in sentence_words:
        lemmatized_words = [word for word, _, _ in lemmatized_with_positions]
        for current_n in combination_sizes:
            if current_n > len(lemmatized_words):
                continue
            if in_sequence:
                combinations = [
                    (tuple(lemmatized_words[i:i + current_n]),
                     [(lemmatized_with_positions[i+j][1], lemmatized_with_positions[i+j][2])
                      for j in range(current_n)])
                    for i in range(len(lemmatized_words) - current_n + 1)
                ]
            else:
                combinations = [
                    (comb, [(lemmatized_with_positions[lemmatized_words.index(w)][1],
                             lemmatized_with_positions[lemmatized_words.index(w)][2])
                            for w in comb])
                    for comb in itertools.combinations(lemmatized_words, current_n)
                ]
            for ngram, indices in combinations:
                occ = occurrences[sentence_idx][ngram]
                occ['count'] += 1
                occ['start_idxs'].append(indices[0][0])
                occ['end_idxs'].append(indices[-1][1])
                occ['sentence_idx'] = sentence_idx
                occ['sentence'] = sentence
                occ['ngram'] = ngram
    result: List[Matched] = []
    for sentence_idx, occ_dict in occurrences.items():
        ngrams = [
            {
                'count': occ['count'],
                'start_idxs': occ['start_idxs'],
                'end_idxs': occ['end_idxs'],
                'sentence_idx': occ['sentence_idx'],
                'sentence': occ['sentence'],
                'ngram': occ['ngram']
            }
            for ngram, occ in occ_dict.items() if occ['count'] >= min_count
        ]
        if ngrams:
            result.append({
                'sentence_idx': sentence_idx,
                'sentence': occ_dict[list(occ_dict.keys())[0]]['sentence'],
                'ngrams': sorted(ngrams, key=lambda x: x['count'], reverse=True)
            })
    return sorted(result, key=lambda x: x['sentence_idx'])


def process_wrapper(text_item: str, n: Optional[int] = None, min_count: int = 1,
                    in_sequence: bool = False, max_n: int = 10) -> List[Matched]:
    return process_single_text(text_item, n, min_count, in_sequence, max_n)


def get_word_sentence_combination_counts(
    text: Union[str, List[str]],
    n: Optional[int] = None,
    min_count: int = 1,
    in_sequence: bool = False,
    max_n: int = 10,
    show_progress: bool = True
) -> Union[List[Matched], List[List[Matched]]]:
    if isinstance(text, str):
        return process_single_text(text, n, min_count, in_sequence, max_n)
    elif isinstance(text, list):
        max_workers = max(1, multiprocessing.cpu_count() // 2)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            try:
                process_func = partial(
                    process_wrapper, n=n, min_count=min_count, in_sequence=in_sequence, max_n=max_n)
                results = list(tqdm(
                    executor.map(process_func, text),
                    total=len(text),
                    desc="Processing texts",
                    disable=not show_progress
                ))
                return results
            except Exception as e:
                print(f"Error in multiprocessing: {e}")
                raise
    else:
        raise ValueError("Input must be a string or a list of strings")
