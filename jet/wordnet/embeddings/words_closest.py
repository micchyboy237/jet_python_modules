import numpy as np
import json
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors
from typing import List, Dict, Tuple
from instruction_generator.helpers.dataset import load_data, save_data
from instruction_generator.utils.time_utils import time_it
from instruction_generator.utils.logger import logger
from instruction_generator.analyzers.helpers import get_tagalog_word_info_dict, get_english_word_info_dict

tagalog_word_info_dict = get_tagalog_word_info_dict()
english_word_info_dict = get_english_word_info_dict()
tl_word_pos_dict = {word: info['pos']
                    for word, info in tagalog_word_info_dict.items()}
en_word_pos_dict = {word: info['pos']
                    for word, info in english_word_info_dict.items()}


pos_mappings = {
    "PRON": ["PRON"],
    "NOUN": ["NOUN"],
    "VERB": ["VERB", "ADP", "ADV"],
    "ADJ": ["ADJ", "ADV"],
    "ADV": ["ADV"],
    "ADP": ["ADP"]
}


class WordSimilarityCalculator:
    def __init__(self, embedding_file: str):
        self.embedding_model = KeyedVectors.load_word2vec_format(
            embedding_file)

    def preprocess(self, text: str) -> str:
        return text.lower().strip()

    def get_embedding(self, word_pos: str) -> np.ndarray:
        return self.embedding_model[word_pos] if word_pos in self.embedding_model else np.zeros(self.embedding_model.vector_size)

    def calculate_similarity(self, word_vec: np.ndarray, other_vec: np.ndarray) -> float:
        if np.all(word_vec == 0) or np.all(other_vec == 0):
            return -1
        return 1 - cosine(word_vec, other_vec)

    def extract_word_and_pos(self, word_pos_str: str) -> Tuple[str, str]:
        word, pos = word_pos_str.rsplit('/', 1)
        return word, pos

    def find_closest_synonyms(self, tagalog_word: str, word_pos_dataset, n: int = 10) -> List[Dict[str, float]]:
        tagalog_word = self.preprocess(tagalog_word)

        try:
            tagalog_pos = tl_word_pos_dict[tagalog_word]
            tagalog_word_pos = f"{tagalog_word}/{tagalog_pos}"
        except Exception as e:
            logger.error(
                f"Tagalog word '{tagalog_word}' not found in the dataset")
            return []
        try:
            english_phrases = word_pos_dataset[tagalog_word]

            # Check if english_phrases has sufficient data
            min_phrases = 2
            if len(english_phrases) < min_phrases:
                logger.error(
                    f"English phrases for '{tagalog_word}' is less than {min_phrases}")
                return []
        except Exception as e:
            logger.error(
                f"Tagalog word pos '{tagalog_word_pos}' not found in the dataset")
            return []
        scores = []
        tagalog_vec = self.get_embedding(tagalog_word_pos)
        for phrase in english_phrases:
            words = phrase.split()
            for word_pos in words:
                english_word, english_pos = self.extract_word_and_pos(word_pos)
                english_word = self.preprocess(english_word)

                if tagalog_pos in pos_mappings and english_pos in pos_mappings[tagalog_pos]:
                    english_vec = self.get_embedding(word_pos)
                    score = self.calculate_similarity(tagalog_vec, english_vec)
                    if score >= 0:
                        # Check if english_word is already in the scores list
                        word_exists = False
                        for item in scores:
                            if item['word'] == english_word and item['pos'] == english_pos:
                                word_exists = True
                                break

                        if not word_exists:
                            scores.append(
                                {"word": english_word, "pos": english_pos, "score": score})

        # Sort the scores in descending order and get the top n
        top_synonyms = sorted(
            scores, key=lambda x: x['score'], reverse=True)[:n]
        return top_synonyms

    @time_it
    def get_embeddings_synonyms(
        self,
        dataset: Dict[str, List[str]],
        n: int = 10
    ) -> Dict[str, List[Dict[str, float]]]:
        synonyms = {}
        # Changed from vocab.keys() to index_to_key
        embeddings_words = self.embedding_model.index_to_key
        logger.info(f"Embeddings words count: {len(embeddings_words)}")
        for tagalog_word_pos in embeddings_words:
            tagalog_word, _ = self.extract_word_and_pos(tagalog_word_pos)
            closest_synonyms = self.find_closest_synonyms(
                tagalog_word, dataset, n)
            if closest_synonyms:
                synonyms[tagalog_word] = closest_synonyms
        return synonyms


if __name__ == '__main__':
    word_translations_pos_file = 'server/static/models/dost-asti-gpt2/base_model/datasets/base/word_texts/tl_en_word_translations_pos.json'
    vector_size = 1200
    filename = f"tl_en_word2vec_{vector_size}.bin"
    embedding_file = f"instruction_generator/wordnet/embeddings/{filename}"
    synonyms_results_file = "instruction_generator/wordnet/embeddings/tl_en_synonyms.json"

    dataset = load_data(word_translations_pos_file)
    calculator = WordSimilarityCalculator(embedding_file)

    word = "atraksyon"
    closest_synonyms = calculator.find_closest_synonyms(word, dataset)
    print(f"Closest synonyms ({len(closest_synonyms)}) for {word}:")
    for synonym in closest_synonyms:
        print(
            f"Word: {synonym['word']}, POS: {synonym['pos']}, Score: {synonym['score']}")

    embeddings_synonyms = calculator.get_embeddings_synonyms(dataset)
    logger.info(f"Embeddings synonyms count: {len(embeddings_synonyms)}")
    if embeddings_synonyms:
        # Sort embeddings_synonyms by keys alphabetically in ascending order
        embeddings_synonyms = dict(
            sorted(embeddings_synonyms.items(), key=lambda item: item[0]))
        save_data(synonyms_results_file, embeddings_synonyms, write=True)
        logger.info(
            f"Saved top synonyms for {len(embeddings_synonyms)} words to {synonyms_results_file}")
