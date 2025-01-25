from jet.llm.query.cleaners import group_and_merge_texts_by_file_name
from llama_index.core.schema import TextNode
import unittest


# Unit Tests

class TestGroupAndMergeTexts(unittest.TestCase):

    def test_single_file_no_overlap(self):
        nodes = [
            TextNode(text="Header 1\nContent 1",
                     metadata={"file_name": "file1.md"}),
            TextNode(text="Header 2\nContent 2",
                     metadata={"file_name": "file1.md"}),
        ]
        expected = {
            "file1.md": "Header 1\nContent 1\nHeader 2\nContent 2"
        }
        self.assertEqual(group_and_merge_texts_by_file_name(nodes), expected)

    def test_single_file_with_overlap(self):
        nodes = [
            TextNode(text="Header 1\nContent 1",
                     metadata={"file_name": "file1.md"}),
            TextNode(text="Content 1\nHeader 2\nContent 2",
                     metadata={"file_name": "file1.md"}),
        ]
        expected = {
            "file1.md": "Header 1\nContent 1\nHeader 2\nContent 2"
        }
        self.assertEqual(group_and_merge_texts_by_file_name(nodes), expected)

    def test_multiple_files(self):
        nodes = [
            TextNode(text="Header 1\nContent 1",
                     metadata={"file_name": "file1.md"}),
            TextNode(text="Header 2\nContent 2",
                     metadata={"file_name": "file1.md"}),
            TextNode(text="Header A\nContent A",
                     metadata={"file_name": "file2.md"}),
            TextNode(text="Header B\nContent B",
                     metadata={"file_name": "file2.md"}),
        ]
        expected = {
            "file1.md": "Header 1\nContent 1\nHeader 2\nContent 2",
            "file2.md": "Header A\nContent A\nHeader B\nContent B"
        }
        self.assertEqual(group_and_merge_texts_by_file_name(nodes), expected)

    def test_unknown_file_name(self):
        nodes = [
            TextNode(text="Header 1\nContent 1", metadata={}),
            TextNode(text="Header 2\nContent 2", metadata={}),
        ]
        expected = {
            "unknown": "Header 1\nContent 1\nHeader 2\nContent 2"
        }
        self.assertEqual(group_and_merge_texts_by_file_name(nodes), expected)


if __name__ == "__main__":
    unittest.main()
