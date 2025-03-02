import unittest
from jet.code.splitter_markdown_utils import get_flat_header_list, get_header_contents, collect_nodes_full_content


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


if __name__ == "__main__":
    unittest.main()
