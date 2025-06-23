import unittest
from typing import List
from jet.scrapers.utils import TreeNode, extract_by_heading_hierarchy


class TestExtractByHeadingHierarchy(unittest.TestCase):
    def setUp(self):
        self.html = """
        <html>
            <body>
                <h1>Title H1</h1>
                <p>Paragraph under H1</p>
                <h2>Subsection H2 A</h2>
                <p>Paragraph under H2 A</p>
                <h3>Subsubsection H3 A.1</h3>
                <p>Paragraph under H3 A.1</p>
                <h2>Subsection H2 B</h2>
                <p>Paragraph under H2 B</p>
                <h3>Subsubsection H3 B.1</h3>
                <p>Paragraph under H3 B.1</p>
                <h4>Deep section H4 B.1.1</h4>
                <p>Paragraph under H4 B.1.1</p>
                <h2>Subsection H2 C</h2>
                <p>Paragraph under H2 C</p>
            </body>
        </html>
        """
        self.trees: List[TreeNode] = extract_by_heading_hierarchy(self.html)

    def test_number_of_headings(self):
        self.assertEqual(len(self.trees), 7)

    def test_tag_order(self):
        tags = [node.tag for node in self.trees]
        expected_tags = ["h1", "h2", "h3", "h2", "h3", "h4", "h2"]
        self.assertEqual(tags, expected_tags)

    def test_content_integrity(self):
        samples = [node.get_content() for node in self.trees]
        expected_contents = [
            "Title H1Paragraph under H1",
            "Subsection H2 AParagraph under H2 A",
            "Subsubsection H3 A.1Paragraph under H3 A.1",
            "Subsection H2 BParagraph under H2 B",
            "Subsubsection H3 B.1Paragraph under H3 B.1",
            "Deep section H4 B.1.1Paragraph under H4 B.1.1",
            "Subsection H2 CParagraph under H2 C",
        ]
        for i, (sample, expected) in enumerate(zip(samples, expected_contents)):
            with self.subTest(i=i):
                self.assertEqual(sample, expected)

    def test_heading_depths(self):
        expected_depths = [0, 1, 2, 1, 2, 3, 1]
        actual_depths = [node.depth for node in self.trees]
        self.assertEqual(actual_depths, expected_depths)

    def test_parent_child_relationships(self):
        id_map = {node.id: node for node in self.trees}

        # Ensure parent IDs are valid
        for node in self.trees:
            if node.parent:
                self.assertIn(node.parent, id_map)

        # Check that all children have their parent's ID in their .parent field
        for node in self.trees:
            for child in node.children:
                self.assertEqual(child.parent, node.id)

    def test_child_ids_unique_and_preserved(self):
        seen_ids = set()

        def collect_ids(node: TreeNode):
            self.assertNotIn(node.id, seen_ids)
            seen_ids.add(node.id)
            for child in node.children:
                collect_ids(child)

        for node in self.trees:
            collect_ids(node)

    def test_get_content_recursive(self):
        """
        Ensures get_content returns combined text from self + all descendants.
        Uses clear result and expected variables for better clarity.
        """
        def _collect_texts_recursively(node: TreeNode) -> List[str]:
            texts = [node.text] if node.text else []
            for child in node.children:
                texts.extend(_collect_texts_recursively(child))
            return texts

        for i, node in enumerate(self.trees):
            expected = "".join(_collect_texts_recursively(node)).strip()
            result = node.get_content()

            with self.subTest(i=i, heading=node.text):
                self.assertEqual(result, expected,
                                 f"Mismatch at heading {node.text}")
                self.assertTrue(result.startswith(node.text or ""),
                                f"Expected result to start with {node.text}")
                self.assertGreaterEqual(len(result), len(
                    node.text or ""), "Content length should include children")


if __name__ == '__main__':
    unittest.main()
