from gensim.models import Doc2Vec
from gensim.models.translation_matrix import BackMappingTranslationMatrix


class Doc2VecTranslator:
    def __init__(self, source_model_path, target_model_path):
        self.source_model = Doc2Vec.load(source_model_path)
        self.target_model = Doc2Vec.load(target_model_path)
        self.translation_matrix = None

    def train_translation_matrix(self, data):
        self.translation_matrix = BackMappingTranslationMatrix(
            self.source_model, self.target_model)
        self.translation_matrix.train(data)

    def infer_vector(self, document_id):
        return self.translation_matrix.infer_vector(self.target_model.dv[document_id])

# Example usage:
# data = read_sentiment_docs(...)[:5]  # Prepare your data
# translator = Doc2VecTranslator(datapath("small_tag_doc_5_iter50"), datapath("large_tag_doc_10_iter50"))
# translator.train_translation_matrix(data)
# result = translator.infer_vector(data[3].tags)
