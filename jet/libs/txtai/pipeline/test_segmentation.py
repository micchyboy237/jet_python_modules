import unittest
from unittest.mock import patch

from jet.libs.txtai.pipeline.segmentation import Segmentation


class TestSegmentation(unittest.TestCase):
    def test_segment_sentences(self):
        with patch("segmentation.sent_tokenize", return_value=["This is the first sentence in a longer paragraph. It is followed by another sentence that continues the thought.", "Here is another distinct sentence that provides more information on the topic at hand."]):
            segmenter = Segmentation(sentences=True)
            text = "This is the first sentence in a longer paragraph. It is followed by another sentence that continues the thought. Here is another distinct sentence that provides more information on the topic at hand."
            result = segmenter(text)
            self.assertEqual(result, [
                "This is the first sentence in a longer paragraph.",
                "It is followed by another sentence that continues the thought.",
                "Here is another distinct sentence that provides more information on the topic at hand."
            ])

    def test_segment_lines(self):
        segmenter = Segmentation(lines=True)
        text = "This is the first line in a structured document.\nThis is the second line providing more context.\nFinally, here is a third line adding additional insights."
        result = segmenter(text)
        self.assertEqual(result, ["This is the first line in a structured document.",
                         "This is the second line providing more context.", "Finally, here is a third line adding additional insights."])

    def test_segment_paragraphs(self):
        segmenter = Segmentation(paragraphs=True)
        text = "This is the first paragraph. It contains multiple sentences to demonstrate segmentation.\n\nThis is the second paragraph, which should be recognized as a separate entity by the segmentation module."
        result = segmenter(text)
        self.assertEqual(result, ["This is the first paragraph. It contains multiple sentences to demonstrate segmentation.",
                         "This is the second paragraph, which should be recognized as a separate entity by the segmentation module."])

    def test_segment_sections(self):
        segmenter = Segmentation(sections=True)
        text = "First section starts here and contains multiple details.\n\n\nSecond section introduces new information and should be properly identified."
        result = segmenter(text)
        self.assertEqual(result, ["First section starts here and contains multiple details.",
                         "Second section introduces new information and should be properly identified."])

    def test_minlength_filtering(self):
        segmenter = Segmentation(lines=True, minlength=15)
        text = "Short\nThis sentence meets the minimum length requirement."
        result = segmenter(text)
        self.assertEqual(
            result, ["This sentence meets the minimum length requirement."])

    def test_text_cleaning(self):
        segmenter = Segmentation()
        text = "  This    sentence   has   irregular   spacing.  "
        result = segmenter(text)
        self.assertEqual(result, "This sentence has irregular spacing.")

    def test_join_enabled(self):
        segmenter = Segmentation(lines=True, join=True)
        text = "This is line one.\nThis is line two continuing the thought."
        result = segmenter(text)
        self.assertEqual(
            result, "This is line one. This is line two continuing the thought.")

    def test_call_with_list_input(self):
        segmenter = Segmentation(lines=True)
        text_list = ["First block of text.\nIt has two lines.",
                     "Second block of text.\nAnother line follows."]
        result = segmenter(text_list)
        expected = [["First block of text.", "It has two lines."],
                    ["Second block of text.", "Another line follows."]]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
