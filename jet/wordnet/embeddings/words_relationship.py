import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from jet.logger import logger, time_it


class WordEmbeddingsRelationship:
    def __init__(self, embeddings_file):
        self.embeddings_dict = self.load_embeddings(embeddings_file)

    @staticmethod
    def load_embeddings(embeddings_file):
        """Load word embeddings from a file."""
        from instruction_generator.helpers.dataset import load_data
        embeddings_dict = load_data(embeddings_file)

        # Convert embedding strings to numpy arrays
        for word, embedding_str in embeddings_dict.items():
            embedding_list = [float(num_str)
                              for num_str in embedding_str.split()]
            embeddings_dict[word] = np.array(embedding_list).reshape(1, -1)
        return embeddings_dict

    def get_embedding(self, word):
        """Get the embedding for a given word."""
        return self.embeddings_dict.get(word)

    def compare_embeddings(self, word, compare_word1, compare_word2):
        """Find a word such that word is to compare_word1 as compare_word2 is to ?"""
        word = word.lower()
        return self.analogy(word, compare_word1, compare_word2)

    @time_it
    def vector_arithmetic(self, word1, word2, word3):
        """Performs vector arithmetic: word1 - word2 + word3"""
        embedding1 = self.get_embedding(word1)
        embedding2 = self.get_embedding(word2)
        embedding3 = self.get_embedding(word3)

        if embedding1 is not None and embedding2 is not None and embedding3 is not None:
            return embedding1 - embedding2 + embedding3
        else:
            missing_words = [word for word in [word1, word2,
                                               word3] if self.get_embedding(word) is None]
            raise ValueError(
                f"Embeddings not found for words: {', '.join(missing_words)}")

    @time_it
    def find_closest_word(self, vector):
        """Finds the word whose embedding is closest to the given vector."""
        closest_word = None
        min_distance = float('inf')

        for word, embedding in self.embeddings_dict.items():
            distance = np.linalg.norm(embedding - vector)
            if distance < min_distance:
                min_distance = distance
                closest_word = word

        return closest_word

    def analogy(self, word1, word2, word3):
        """Finds word4 such that word1 is to word2 as word3 is to word4"""
        result_vector = self.vector_arithmetic(word1, word2, word3)
        return self.find_closest_word(result_vector)


# Example usage
embedding_file = 'data/samples.json'
relationship = WordEmbeddingsRelationship(embedding_file)
result_word = relationship.compare_embeddings('data', 'philippine', 'data')
print(
    f"Completing the analogy: 'area' is to 'philippine' as 'data' is to '{result_word}'")
