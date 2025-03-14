import psutil
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from instruction_generator.utils.logger import logger
from instruction_generator.utils.time_utils import time_it


class WordEmbeddings:
    _model = None  # Class attribute to store the loaded model

    def __init__(self, embeddings_file):
        if WordEmbeddings._model is None:
            logger.info(f"Loading embeddings from {embeddings_file}...")
            WordEmbeddings._model = self.load_embeddings(
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
            return WordEmbeddings._model[word].reshape(1, -1)
        except KeyError:
            return None

    def predict_next_word(self, sentence_start):
        sentence_start = sentence_start.lower()
        words = sentence_start.split()
        last_word = words[-1] if words else None
        sentence_vector = np.zeros(self._model.vector_size)

        known_words = 0
        for word in words:
            if word in self._model:
                sentence_vector += self.get_embedding(word).flatten()
                known_words += 1

        if known_words > 0:
            sentence_vector /= known_words  # Averaging the vectors

        # Exclude last word in the similarity check
        if last_word:
            most_similar = [word for word, similarity in self._model.similar_by_vector(sentence_vector, topn=10)
                            if word != last_word]
        else:
            most_similar = self._model.similar_by_vector(
                sentence_vector, topn=1)

        return most_similar[0] if most_similar else None


def main():
    embedding_file = 'data/taglish-word2vec-cbow-100.bin'
    comparer = WordEmbeddings(embedding_file)

    while True:
        text = input("Enter initial text (or 'exit' to quit): ")
        # quit on 'exit' or ctrl+c
        if text == 'exit' or text == KeyboardInterrupt:
            logger.info("Exiting...")
            break

        result = comparer.predict_next_word(text)

        print(f"Predicted next word: {result}")


if __name__ == "__main__":
    main()
