from instruction_generator.utils.time_utils import time_it
from gensim.models import Word2Vec
from instruction_generator.wordnet.gensim_scripts.datasets import get_translation_sentences, get_translation_word_synonyms
from instruction_generator.helpers.words import get_words
from instruction_generator.utils.logger import logger
import os


class WordSentenceEmbeddings:
    @time_it
    def __init__(self, model_path, sentences, *args, **kwargs):
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = self.load_model(model_path, *args, **kwargs)
        else:
            print("Preprocessing sentences")
            lower_sentences = []
            for sentence in sentences:
                words = get_words(sentence)
                words = [word.lower() for word in words]
                lower_sentences.append(words)
            print("Creating new model")
            self.model = Word2Vec(
                lower_sentences,
                *args,
                window=8,
                **kwargs)
            print(f"Saving model to {model_path}")
            self.save_model(model_path)

    def __getattr__(self, name):
        """
        Delegate attribute lookup to self.model.

        This method is called if the attribute `name` isn't found in the
        PhraseDetector instance. If `name` is a method or attribute of self.model,
        it returns that method/attribute. Otherwise, it raises an AttributeError.
        """
        # Check if the attribute exists in self.model and return it.
        # This allows direct access to methods and properties of self.model.
        try:
            return getattr(self.model, name)
        except AttributeError:
            # If the attribute is not found in self.model, raise an AttributeError
            # to signal that this object doesn't have the requested attribute.
            raise AttributeError(
                f"'Word2Vec' object has no attribute '{name}'")

    @time_it
    def load_model(self, model_path, *args, **kwargs) -> Word2Vec:
        model = Word2Vec.load(model_path, *args, **kwargs)
        return model

    @time_it
    def save_model(self, model_path):
        self.model.save(model_path)


if __name__ == '__main__':
    model_path = 'instruction_generator/wordnet/embeddings/gensim_jet_sentence_model.pkl'
    # sentences = []
    # synonyms = get_translation_word_synonyms()

    # for word, syns in synonyms.items():
    #     sym_words = [word] + syns
    #     sentences.append(" ".join(sym_words))
    sentences = get_translation_sentences()

    word2vec = WordSentenceEmbeddings(model_path, sentences)
    model = word2vec.model

    topn = 50

    while True:
        base_word = input("Enter any word: ")
        # quit on 'exit' or ctrl+c
        if base_word == 'exit' or base_word == KeyboardInterrupt:
            logger.info("Exiting...")
            break

        try:
            results = model.wv.most_similar(
                base_word, topn=topn)
            print(f"Most similar words to '{base_word}':\n{results}")
        except KeyError:
            print(f"'{base_word}' not found in the model.")
