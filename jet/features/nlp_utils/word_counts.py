from collections import Counter, defaultdict
import math
from typing import Literal, Tuple, Union, List, Dict, Optional, TypedDict
from jet.features.nlp_utils.utils import get_wordnet_pos
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from jet.features.nlp_utils.nlp_types import WordOccurrence
from jet.wordnet.sentence import split_sentences


def get_word_counts_lemmatized(
    text: Union[str, List[str]],
    pos: Optional[List[Literal['noun', 'verb',
                               'adjective', 'adverb', 'number']]] = None,
    min_count: int = 1,
    as_score: bool = False,
    percent_threshold: float = 0.0
) -> Union[Dict[str, List[WordOccurrence]], List[Dict[str, List[WordOccurrence]]]]:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    pos_mapping = {
        'noun': 'N', 'verb': 'V', 'adjective': 'J', 'adverb': 'R', 'number': 'CD'
    }

    def process_single_text(single_text: str) -> List[Tuple[str, str, int, int, int, str]]:
        sentences = split_sentences(single_text)
        result = []
        char_offset = 0

        for sentence_idx, sentence in enumerate(sentences):
            tokens = word_tokenize(sentence.lower())
            words = [(token, i) for i, token in enumerate(tokens) if (token.isalpha() or token.isdigit())
                     and token not in stop_words]
            tagged_words = pos_tag([w[0] for w in words])
            word_positions = []
            current_pos = char_offset
            for word, _ in words:
                start_idx = single_text[current_pos:].lower().find(
                    word) + current_pos
                end_idx = start_idx + len(word)
                word_positions.append((word, start_idx, end_idx))
                current_pos = end_idx

            for (word, idx), (word_tagged, tag), (_, start_idx, end_idx) in zip(words, tagged_words, word_positions):
                wordnet_pos = get_wordnet_pos(tag)
                lemmatized_word = lemmatizer.lemmatize(word, pos=wordnet_pos)
                if pos is None or any(tag.startswith(pos_mapping[p]) for p in pos):
                    result.append((lemmatized_word, tag, start_idx,
                                  end_idx, sentence_idx, sentence))

            char_offset += len(sentence) + 1

        return result

    def calculate_scores(occurrences: List[WordOccurrence], total_words: int) -> List[WordOccurrence]:
        if not occurrences:
            return []
        counts = Counter(occ['word'] for occ in occurrences)
        raw_scores = {
            word: count * (1 + math.log(len(word)))
            for word, count in counts.items()
        }
        max_score = max(raw_scores.values(), default=1.0)
        score_map = {word: (score / max_score) * 100 for word,
                     score in raw_scores.items()}
        result = []
        for occ in occurrences:
            new_occ: WordOccurrence = {
                'score': score_map[occ['word']],
                'start_idxs': occ['start_idxs'],
                'end_idxs': occ['end_idxs'],
                'sentence_idx': occ['sentence_idx'],
                'word': occ['word'],
                'sentence': occ['sentence']
            }
            result.append(new_occ)
        return result

    if isinstance(text, str):
        lemmatized_words = process_single_text(text)
        total_words = len(lemmatized_words)
        occurrences: Dict[str, Dict[int, WordOccurrence]] = defaultdict(lambda: defaultdict(
            lambda: {'count': 0, 'start_idxs': [], 'end_idxs': [], 'sentence_idx': 0, 'word': '', 'sentence': ''}))
        for word, _, start_idx, end_idx, sentence_idx, sentence in lemmatized_words:
            occ = occurrences[word][sentence_idx]
            occ['count'] += 1
            occ['start_idxs'].append(start_idx)
            occ['end_idxs'].append(end_idx)
            occ['sentence_idx'] = sentence_idx
            occ['word'] = word
            occ['sentence'] = sentence

        min_count_threshold = max(min_count, int(
            total_words * percent_threshold / 100.0))
        filtered_occurrences = {}
        for word, occ_dict in occurrences.items():
            occ_list = [occ for occ in occ_dict.values(
            ) if occ['count'] >= min_count_threshold]
            if occ_list:
                filtered_occurrences[word] = sorted(
                    occ_list, key=lambda x: x['sentence_idx'])

        if as_score and total_words > 0:
            result = {}
            for word, occ_list in filtered_occurrences.items():
                result[word] = calculate_scores(occ_list, total_words)
            return dict(sorted(result.items(), key=lambda x: max(o.get('score', 0) for o in x[1]), reverse=True))

        return dict(sorted(filtered_occurrences.items(), key=lambda x: sum(o['count'] for o in x[1]), reverse=True))

    elif isinstance(text, list):
        all_words_by_text = [process_single_text(
            single_text) for single_text in text]
        all_words = [
            item for text_words in all_words_by_text for item in text_words]
        total_words = len(all_words)
        total_counts = Counter(word for word, _, _, _, _, _ in all_words)
        min_count_threshold = max(min_count, int(
            total_words * percent_threshold / 100.0))
        valid_words = {word for word, count in total_counts.items()
                       if count >= min_count_threshold}

        result = []
        for text_words in all_words_by_text:
            occurrences: Dict[str, Dict[int, WordOccurrence]] = defaultdict(lambda: defaultdict(
                lambda: {'count': 0, 'start_idxs': [], 'end_idxs': [], 'sentence_idx': 0, 'word': '', 'sentence': ''}))
            for word, _, start_idx, end_idx, sentence_idx, sentence in text_words:
                if word in valid_words:
                    occ = occurrences[word][sentence_idx]
                    occ['count'] += 1
                    occ['start_idxs'].append(start_idx)
                    occ['end_idxs'].append(end_idx)
                    occ['sentence_idx'] = sentence_idx
                    occ['word'] = word
                    occ['sentence'] = sentence
            total_words_per_text = len(text_words)
            filtered_occurrences = {}
            for word, occ_dict in occurrences.items():
                occ_list = [occ for occ in occ_dict.values(
                ) if occ['count'] >= min_count_threshold]
                if occ_list:
                    filtered_occurrences[word] = sorted(
                        occ_list, key=lambda x: x['sentence_idx'])
            if as_score and total_words_per_text > 0:
                filtered_occurrences = {word: calculate_scores(occ_list, total_words_per_text)
                                        for word, occ_list in filtered_occurrences.items()}
            result.append(dict(sorted(filtered_occurrences.items(),
                                      key=lambda x: sum(
                                          o.get('count', o.get('score', 0)) for o in x[1]),
                                      reverse=True)))
        return result

    else:
        raise TypeError("Input must be a string or a list of strings")
