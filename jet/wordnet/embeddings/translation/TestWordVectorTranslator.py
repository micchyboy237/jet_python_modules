import unittest
from unittest.mock import MagicMock, patch
# Ensure you import your actual class
from instruction_generator.wordnet.embeddings.translation.WordVectorTranslator import WordVectorTranslator


class TestWordVectorTranslator(unittest.TestCase):
    def setUp(self):
        # Mock paths for English and Tagalog models
        self.mock_en_model_path = "path/to/english/model"
        self.mock_tl_model_path = "path/to/tagalog/model"

        # Sample word pairs for translation matrix fitting
        self.word_pairs = [
            ("one", "isa"), ("two", "dalawa"), ("three", "tatlo"),
            ("hello", "kamusta"), ("thank you", "salamat")
        ]

    @patch('WordVectorTranslator.KeyedVectors.load_word2vec_format')
    def test_initialization(self, mock_load_word2vec_format):
        # Mock the load_word2vec_format to return a MagicMock object
        mock_load_word2vec_format.return_value = MagicMock()
        translator = WordVectorTranslator(
            self.mock_en_model_path, self.mock_tl_model_path)

        # Assert that models are loaded during initialization
        mock_load_word2vec_format.assert_called()
        self.assertIsNotNone(translator.source_model)
        self.assertIsNotNone(translator.target_model)

    @patch('WordVectorTranslator.TranslationMatrix')
    @patch('WordVectorTranslator.KeyedVectors.load_word2vec_format')
    def test_fit_translation_matrix(self, mock_load_word2vec_format, mock_TranslationMatrix):
        # Mock load_word2vec_format and TranslationMatrix to return MagicMock objects
        mock_load_word2vec_format.return_value = MagicMock()
        mock_TranslationMatrix.return_value = MagicMock()

        translator = WordVectorTranslator(
            self.mock_en_model_path, self.mock_tl_model_path)
        translator.fit_translation_matrix(self.word_pairs)

        # Assert that TranslationMatrix is instantiated and trained with word pairs
        mock_TranslationMatrix.assert_called_once_with(
            translator.source_model, translator.target_model, word_pairs=self.word_pairs)
        translator.translation_matrix.train.assert_called_once_with(
            self.word_pairs)

    @patch('WordVectorTranslator.TranslationMatrix.translate')
    @patch('WordVectorTranslator.TranslationMatrix')
    @patch('WordVectorTranslator.KeyedVectors.load_word2vec_format')
    def test_translate(self, mock_load_word2vec_format, mock_TranslationMatrix, mock_translate):
        # Mock translation output
        mock_translate.return_value = {
            "hello": [("kamusta", 0.8)], "one": [("isa", 0.9)]}

        mock_load_word2vec_format.return_value = MagicMock()
        mock_TranslationMatrix.return_value = MagicMock(
            translate=mock_translate)

        translator = WordVectorTranslator(
            self.mock_en_model_path, self.mock_tl_model_path)
        translator.fit_translation_matrix(self.word_pairs)
        result = translator.translate(["hello", "one"], topn=1)

        # Assert translate method is called and returns the expected result
        mock_translate.assert_called_once_with(["hello", "one"], topn=1)
        self.assertEqual(
            result, {"hello": [("kamusta", 0.8)], "one": [("isa", 0.9)]})


if __name__ == '__main__':
    unittest.main()
