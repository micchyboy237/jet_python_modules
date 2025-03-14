from gensim.models import KeyedVectors, TranslationMatrix
from gensim.test.utils import datapath, temporary_file
from instruction_generator.wordnet.gensim_scripts.datasets import get_translation_word_pairs
from instruction_generator.wordnet.gensim_scripts.Embeddings import WordSentenceEmbeddings
from instruction_generator.utils.time_utils import time_it
import os


class WordVectorTranslator:
    def __init__(self, source_model_path, target_model_path, word_pairs):
        if not os.path.exists(source_model_path) or not os.path.exists(target_model_path):
            # get sources and targets from word_pairs
            sources, targets = zip(*word_pairs)

            print("Creating new source model")
            sources_word2vec = WordSentenceEmbeddings(
                source_model_path, sources)
            sources_word2vec.save_model(source_model_path)

            print("Creating new target model")
            targets_word2vec = WordSentenceEmbeddings(
                target_model_path, targets)
            targets_word2vec.save_model(target_model_path)

        self.source_model = KeyedVectors.load_word2vec_format(
            source_model_path)
        self.target_model = KeyedVectors.load_word2vec_format(
            target_model_path)
        self.translation_matrix = None

    @time_it
    def fit_translation_matrix(self, word_pairs):
        self.translation_matrix = TranslationMatrix(
            self.source_model, self.target_model, word_pairs=word_pairs)
        self.translation_matrix.train(word_pairs)

    @time_it
    def translate(self, words, topn=3):
        return self.translation_matrix.translate(words, topn=topn)

    @time_it
    def save_model(self, filename):
        with temporary_file(filename) as fname:
            self.translation_matrix.save(fname)
            return fname

    @time_it
    def load_model(self, filename):
        self.translation_matrix = TranslationMatrix.load(filename)


if __name__ == '__main__':
    source = 'en'
    target = 'tl'
    direction = f'{source}-{target}'
    source_model_path = f'instruction_generator/wordnet/embeddings/gensim_jet_{source}_word_model.pkl'
    target_model_path = f'instruction_generator/wordnet/embeddings/gensim_jet_{target}_word_model.pkl'

    # Paths to source and target model files need to be correctly set
    word_pairs = get_translation_word_pairs(direction)
    translator = WordVectorTranslator(
        source_model_path, target_model_path, word_pairs)

    translator.fit_translation_matrix(word_pairs)

    translation_result = translator.translate(["one", "hello"], topn=3)
    print(translation_result)
