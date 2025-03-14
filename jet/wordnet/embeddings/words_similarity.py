import psutil
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from instruction_generator.utils.logger import logger
from instruction_generator.utils.time_utils import time_it


class WordEmbeddingComparer:
    _model = None  # Class attribute to store the loaded model

    def __init__(self, embeddings_file):
        if WordEmbeddingComparer._model is None:
            logger.info(f"Loading embeddings from {embeddings_file}...")
            WordEmbeddingComparer._model = self.load_embeddings(
                embeddings_file)
            logger.memory("RAM usage after loading embeddings:")
            logger.info("Done loading embeddings.")
        else:
            logger.info("Using previously loaded embeddings.")

    @staticmethod
    @time_it
    def load_embeddings(embeddings_file):
        """Load word embeddings from a file."""
        return KeyedVectors.load_word2vec_format(embeddings_file)

    def get_embedding(self, word):
        """Get the embedding for a given word."""
        try:
            return WordEmbeddingComparer._model[word].reshape(1, -1)
        except KeyError:
            return None

    @time_it
    def compare_embeddings(self, word, compare_words):
        """Compare base word with a list of words and print cosine similarities."""
        word = word.lower()
        logger.memory("RAM usage before comparing embeddings:")
        base_embedding = self.get_embedding(word)
        if base_embedding is None:
            print(f"Embedding not found for base word: '{word}'")
            return

        for word in compare_words:
            compare_embedding = self.get_embedding(word)
            if compare_embedding is not None:
                similarity = cosine_similarity(
                    base_embedding, compare_embedding)
                print(
                    f"Cosine similarity between '{word}' and '{word}': {similarity[0][0]}")
            else:
                print(f"Embedding not found for word: '{word}'")
        logger.memory("RAM usage after comparing embeddings:")


def main():
    embedding_file = 'instruction_generator/wordnet/embeddings/datasets/taglish-word2vec-cbow-100.bin'
    comparer = WordEmbeddingComparer(embedding_file)

    while True:
        base_word = input("Enter the base word (or 'exit' to quit): ")
        # quit on 'exit' or ctrl+c
        if base_word == 'exit' or base_word == KeyboardInterrupt:
            logger.info("Exiting...")
            break

        compare_words = input(
            "Enter words to compare (separated by space): ").split()
        comparer.compare_embeddings(base_word, compare_words)


if __name__ == "__main__":
    main()
