import unittest
from jet.code.splitter_markdown_utils import get_flat_header_list, get_header_contents, collect_nodes_full_content, merge_md_header_contents


class TestGetHeaderContents(unittest.TestCase):
    def setUp(self):
        self.sample_md = """
        # Header 1
        Content under header 1.

        ## Subheader 1.1
        Content under subheader 1.1.

        ### Subheader 1.1.1
        Content under subheader 1.1.1.

        ## Subheader 1.2
        Content under subheader 1.2.

        # Header 2
        Content under header 2.
        """

    def test_flat_structure(self):
        md_text = """
        # Header 1
        Content 1

        ## Header 2
        Content 2

        ### Header 3
        Content 3
        """
        result = get_header_contents(md_text)
        self.assertEqual(len(result), 1)
        self.assertIn('# Header 1', result[0]['header'])
        self.assertIn('## Header 2', result[0]['child_nodes'][0]['header'])
        self.assertIn('### Header 3',
                      result[0]['child_nodes'][0]['child_nodes'][0]['header'])

    def test_nested_hierarchy(self):
        result = get_header_contents(self.sample_md)
        # Two root headers: Header 1 and Header 2
        self.assertEqual(len(result), 2)

        header_1 = result[0]
        self.assertIn('# Header 1', header_1['header'])
        self.assertEqual(len(header_1['child_nodes']), 2)

        subheader_1_1 = header_1['child_nodes'][0]
        self.assertIn('## Subheader 1.1', subheader_1_1['header'])
        self.assertEqual(len(subheader_1_1['child_nodes']), 1)

        subheader_1_1_1 = subheader_1_1['child_nodes'][0]
        self.assertIn('### Subheader 1.1.1', subheader_1_1_1['header'])
        self.assertIn('Content under subheader 1.1.1.',
                      subheader_1_1_1['details'])

        subheader_1_2 = header_1['child_nodes'][1]
        self.assertIn('## Subheader 1.2', subheader_1_2['header'])

        header_2 = result[1]
        self.assertIn('# Header 2', header_2['header'])
        self.assertIn('Content under header 2.', header_2['details'])

    def test_collect_full_content(self):
        # Test when include_child_contents=True
        result = get_header_contents(
            self.sample_md, include_child_contents=True)

        # Test full content for Header 1 and its child nodes
        header_1 = result[0]
        full_content_1 = collect_nodes_full_content(header_1)
        self.assertIn("Content under header 1.", full_content_1)
        self.assertIn("Content under subheader 1.1.", full_content_1)
        self.assertIn("Content under subheader 1.1.1.", full_content_1)
        self.assertIn("Content under subheader 1.2.", full_content_1)

        # Test full content for Header 2
        header_2 = result[1]
        full_content_2 = collect_nodes_full_content(header_2)
        self.assertIn("Content under header 2.", full_content_2)

    def test_skip_heading_levels(self):
        md_text = """
        # Header 1
        Content 1
        
        ### Header 2
        Content 2

        ## Header 3
        Content 3

        ## Header 4
        Content 4

        #### Header 5
        Content 5

        ## Header 6
        Content 6

        ### Header 7
        Content 7
        """

        # Test skipping different header levels
        # Example: Skip level 2 headers (##)
        result = get_header_contents(md_text)

        # We expect headers at level 2 (##) to be skipped
        header_1 = result[0]
        self.assertIn("Header 1", header_1['header'])  # Header 1 is included

        header_1_child_nodes = header_1['child_nodes']
        self.assertEqual(len(header_1_child_nodes), 4)
        self.assertIn("### Header 2", header_1_child_nodes[0]['header'])
        self.assertIn("## Header 3", header_1_child_nodes[1]['header'])

        self.assertIn("## Header 4", header_1_child_nodes[2]['header'])
        header_4_nested_child_nodes = header_1_child_nodes[2]['child_nodes']
        self.assertEqual(len(header_4_nested_child_nodes), 1)
        self.assertIn("#### Header 5",
                      header_4_nested_child_nodes[0]['header'])

        self.assertIn("## Header 6", header_1_child_nodes[3]['header'])
        header_6_nested_child_nodes = header_1_child_nodes[3]['child_nodes']
        self.assertEqual(len(header_6_nested_child_nodes), 1)
        self.assertIn("### Header 7", header_6_nested_child_nodes[0]['header'])

        # Test skipping level 3 headers (###)
        result = get_header_contents(
            md_text, headers_to_split_on=["#", "##", "####"])

        # We expect headers at level 3 (###) to be skipped
        header_2 = result[0]
        self.assertIn("# Header 1", header_2['header'])
        header_2_child_nodes = header_2['child_nodes']
        self.assertEqual(len(header_2_child_nodes), 3)
        self.assertIn("## Header 3", header_2_child_nodes[0]['header'])
        self.assertIn("## Header 4", header_2_child_nodes[1]['header'])
        self.assertIn("## Header 6", header_2_child_nodes[2]['header'])

    def test_flat_header_list(self):
        result = get_header_contents(
            self.sample_md, include_child_contents=True)
        # Check the flat list for the first header (Header 1)
        header_1 = result[0]
        flat_list_1 = get_flat_header_list(header_1)

        self.assertEqual(len(flat_list_1), 4)
        self.assertIn(header_1, flat_list_1)

        header_2 = result[1]
        flat_list_2 = get_flat_header_list(header_2)
        self.assertEqual(len(flat_list_2), 1)

        all_headers = result
        all_flat_list = get_flat_header_list(all_headers)
        self.assertEqual(len(all_flat_list), 5)


class TestHeaderMetadata(unittest.TestCase):
    def setUp(self):
        self.sample_md = """
        # Header 1
        Content under header 1.

        ## Subheader 1.1
        Content under subheader 1.1.

        ### Subheader 1.1.1
        Content under subheader 1.1.1.

        ## Subheader 1.2
        Content under subheader 1.2.

        # Header 2
        Content under header 2.
        """

    def test_metadata_values(self):
        result = get_header_contents(self.sample_md)

        # Check metadata for Header 1
        header_1 = result[0]
        self.assertEqual(header_1['metadata']['start_line_idx'], 1)
        self.assertEqual(header_1['metadata']['depth'], 1)

        # Check metadata for Subheader 1.1
        subheader_1_1 = header_1['child_nodes'][0]
        self.assertEqual(subheader_1_1['metadata']['start_line_idx'], 4)
        self.assertEqual(subheader_1_1['metadata']['depth'], 2)

        # Check metadata for Subheader 1.1.1
        subheader_1_1_1 = subheader_1_1['child_nodes'][0]
        self.assertEqual(subheader_1_1_1['metadata']['start_line_idx'], 7)
        self.assertEqual(subheader_1_1_1['metadata']['depth'], 3)

        # Check metadata for Subheader 1.2
        subheader_1_2 = header_1['child_nodes'][1]
        self.assertEqual(subheader_1_2['metadata']['start_line_idx'], 10)
        self.assertEqual(subheader_1_2['metadata']['depth'], 2)

        # Check metadata for Header 2
        header_2 = result[1]
        self.assertEqual(header_2['metadata']['start_line_idx'], 13)
        self.assertEqual(header_2['metadata']['depth'], 1)

    def test_metadata_end_line_idx(self):
        result = get_header_contents(self.sample_md)

        # Ensure end_line_idx covers the full content range
        header_1 = result[0]
        self.assertEqual(header_1['metadata']['end_line_idx'], 4)

        subheader_1_1 = header_1['child_nodes'][0]
        self.assertEqual(subheader_1_1['metadata']['end_line_idx'], 7)

        subheader_1_1_1 = subheader_1_1['child_nodes'][0]
        self.assertEqual(subheader_1_1_1['metadata']['end_line_idx'], 10)

        subheader_1_2 = header_1['child_nodes'][1]
        self.assertEqual(subheader_1_2['metadata']['end_line_idx'], 13)

        header_2 = result[1]
        self.assertEqual(header_2['metadata']['end_line_idx'], 16)

    def test_metadata_values_with_child_contents(self):
        result = get_header_contents(
            self.sample_md, include_child_contents=True)

        # Check metadata for Header 1
        header_1 = result[0]
        self.assertEqual(header_1['metadata']['start_line_idx'], 1)
        self.assertEqual(header_1['metadata']['depth'], 1)

        # Check metadata for Subheader 1.1
        subheader_1_1 = header_1['child_nodes'][0]
        self.assertEqual(subheader_1_1['metadata']['start_line_idx'], 4)
        self.assertEqual(subheader_1_1['metadata']['depth'], 2)

        # Check metadata for Subheader 1.1.1
        subheader_1_1_1 = subheader_1_1['child_nodes'][0]
        self.assertEqual(subheader_1_1_1['metadata']['start_line_idx'], 7)
        self.assertEqual(subheader_1_1_1['metadata']['depth'], 3)

        # Check metadata for Subheader 1.2
        subheader_1_2 = header_1['child_nodes'][1]
        self.assertEqual(subheader_1_2['metadata']['start_line_idx'], 10)
        self.assertEqual(subheader_1_2['metadata']['depth'], 2)

        # Check metadata for Header 2
        header_2 = result[1]
        self.assertEqual(header_2['metadata']['start_line_idx'], 13)
        self.assertEqual(header_2['metadata']['depth'], 1)

        # Check end_line_idx for Header 1 with child content
        self.assertEqual(header_1['metadata']['end_line_idx'], 12)

        # Check end_line_idx for Subheader 1.1 with child content
        self.assertEqual(subheader_1_1['metadata']['end_line_idx'], 9)

        # Check end_line_idx for Subheader 1.1.1 with child content
        self.assertEqual(subheader_1_1_1['metadata']['end_line_idx'], 9)

        # Check end_line_idx for Subheader 1.2 with child content
        self.assertEqual(subheader_1_2['metadata']['end_line_idx'], 12)

        # Check end_line_idx for Header 2 with child content
        self.assertEqual(header_2['metadata']['end_line_idx'], 15)

    def test_hierarchy_depth(self):
        result = get_header_contents(self.sample_md)

        header_1 = result[0]
        self.assertEqual(header_1['metadata']['depth'], 1)
        self.assertEqual(header_1['child_nodes'][0]['metadata']['depth'], 2)
        self.assertEqual(header_1['child_nodes'][0]
                         ['child_nodes'][0]['metadata']['depth'], 3)
        self.assertEqual(header_1['child_nodes'][1]['metadata']['depth'], 2)

        header_2 = result[1]
        self.assertEqual(header_2['metadata']['depth'], 1)


# Mock tokenizer function (simulating a token count for a string)
def mock_tokenizer(text: str) -> list[str]:
    return text.split()  # Tokenize by splitting on spaces (simplified)


class TestMergeMdHeaderContents(unittest.TestCase):

    def setUp(self):
        """Setup common test data"""
        self.header_contents = [
            {"content": "# Header 1\nContent 1",
                "length": 5, "header": "# Header 1"},
            {"content": "## Subheader 1\nMore content",
                "length": 4, "header": "## Subheader 1"},
            {"content": "### Subheader 2\nEven more content",
                "length": 5, "header": "### Subheader 2"},
            {"content": "## Subheader 3\nFinal content",
                "length": 4, "header": "## Subheader 3"},
        ]

    def test_basic_merge(self):
        """Test merging within max_tokens limit"""
        result = merge_md_header_contents(
            self.header_contents, max_tokens=20, tokenizer=mock_tokenizer)
        self.assertEqual(len(result), 1)
        self.assertIn("Content 1", result[0]["content"])
        self.assertIn("Final content", result[0]["content"])

    def test_split_at_max_tokens(self):
        """Ensure splitting happens when max_tokens is exceeded"""
        result = merge_md_header_contents(
            self.header_contents, max_tokens=10, tokenizer=mock_tokenizer)
        self.assertGreater(len(result), 1)
        self.assertTrue(all(chunk["length"] <= 10 for chunk in result))

    def test_respects_min_tokens(self):
        """Ensure min_tokens is respected and avoids small chunks"""
        result = merge_md_header_contents(
            self.header_contents, max_tokens=15, min_tokens=8, tokenizer=mock_tokenizer)
        self.assertTrue(all(chunk["length"] >= 8 for chunk in result))

    def test_exact_max_tokens(self):
        """Ensure exact max_tokens is handled correctly"""
        exact_content = [
            {"content": "# Header A\nword " * 5,
                "length": 5, "header": "# Header A"},
            {"content": "## Header B\nword " * 5,
                "length": 5, "header": "## Header B"},
        ]
        result = merge_md_header_contents(
            exact_content, max_tokens=10, tokenizer=mock_tokenizer)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["length"], 10)

    def test_large_content_splits_properly(self):
        """Ensure a single large content block is split correctly"""
        large_content = [
            {"content": "# Large Header\n" + "word " * 50,
                "length": 50, "header": "# Large Header"},
        ]
        result = merge_md_header_contents(
            large_content, max_tokens=20, tokenizer=mock_tokenizer)
        self.assertGreater(len(result), 1)
        self.assertTrue(all(chunk["length"] <= 20 for chunk in result))

    def test_multiple_small_contents_merge_efficiently(self):
        """Ensure multiple small headers merge into a single chunk"""
        small_contents = [
            {"content": f"# Header {i}\nword",
                "length": 1, "header": f"# Header {i}"}
            for i in range(10)
        ]
        result = merge_md_header_contents(
            small_contents, max_tokens=10, tokenizer=mock_tokenizer)
        self.assertEqual(len(result), 1)


if __name__ == '__main__':
    unittest.main()
