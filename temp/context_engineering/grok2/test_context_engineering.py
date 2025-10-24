import unittest
from unittest.mock import patch, MagicMock
from context_engineer import summarize_document, rank_documents, engineer_context, generate_response

class TestContextEngineering(unittest.TestCase):

    def test_summarize_document_short(self):
        doc = "Short text."
        self.assertEqual(summarize_document(doc), "Short text.")

    @patch('openai.ChatCompletion.create')
    def test_summarize_document_long(self, mock_create):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Summary"))]
        mock_create.return_value = mock_response
        doc = "A" * 600
        self.assertEqual(summarize_document(doc), "Summary")

    def test_rank_documents(self):
        query = "AI context"
        docs = ["AI is great.", "Context engineering in AI.", "Random text."]
        ranked = rank_documents(query, docs)
        self.assertEqual(ranked[0], "Context engineering in AI.")

    def test_engineer_context(self):
        query = "What is AI?"
        docs = ["AI definition.", "Machine learning.", "Deep learning."]
        context = engineer_context(query, docs)
        self.assertIn("Document 1: AI definition.", context)

    @patch('openai.ChatCompletion.create')
    def test_generate_response(self, mock_create):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Generated answer"))]
        mock_create.return_value = mock_response
        query = "Test query"
        docs = ["Doc1", "Doc2"]
        result = generate_response(query, docs)
        self.assertEqual(result, "Generated answer")
        # Check if prompt was called with engineered context
        called_prompt = mock_create.call_args[1]['messages'][0]['content']
        self.assertIn("<relevant_context>", called_prompt)
        self.assertIn("Doc1", called_prompt)

if __name__ == '__main__':
    unittest.main()