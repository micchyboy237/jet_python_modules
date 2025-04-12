from pprint import pprint
import os
import unittest
from jet.scrapers.utils import extract_clickable_texts_from_rendered_page, extract_element_screenshots, extract_form_elements, extract_search_inputs, extract_title_and_metadata, extract_internal_links, extract_by_heading_hierarchy


class TestExtractTitleAndMetadata(unittest.TestCase):

    def test_basic_title_and_meta_extraction(self):
        sample = """
            <html>
                <head>
                    <title>Example Page</title>
                    <meta name="description" content="This is a test page.">
                    <meta property="og:title" content="OpenGraph Title">
                </head>
                <body></body>
            </html>
        """
        expected = {
            "title": "Example Page",
            "metadata": {
                "description": "This is a test page.",
                "og:title": "OpenGraph Title"
            }
        }
        result = extract_title_and_metadata(sample)
        self.assertEqual(result, expected)

    def test_missing_title(self):
        sample = """
            <html><head>
                <meta name="keywords" content="python, test">
            </head></html>
        """
        expected = {
            "title": "",
            "metadata": {
                "keywords": "python, test"
            }
        }
        result = extract_title_and_metadata(sample)
        self.assertEqual(result, expected)

    def test_meta_without_name_or_property(self):
        sample = """
            <html><head>
                <meta content="No key here">
                <meta name="author" content="John Doe">
            </head></html>
        """
        expected = {
            "title": "",
            "metadata": {
                "author": "John Doe"
            }
        }
        result = extract_title_and_metadata(sample)
        self.assertEqual(result, expected)

    def test_meta_with_duplicate_keys(self):
        sample = """
            <html><head>
                <title>Dup Test</title>
                <meta name="author" content="John">
                <meta name="author" content="Jane">
            </head></html>
        """
        expected = {
            "title": "Dup Test",
            "metadata": {
                "author": "Jane"  # Last one wins
            }
        }
        result = extract_title_and_metadata(sample)
        self.assertEqual(result, expected)

    def test_meta_with_empty_content(self):
        sample = """
            <html><head>
                <meta name="description" content="">
                <meta property="og:type" content="website">
            </head></html>
        """
        expected = {
            "title": "",
            "metadata": {
                "og:type": "website"
            }
        }
        result = extract_title_and_metadata(sample)
        self.assertEqual(result, expected)


class TestExtractInternalLinks(unittest.TestCase):
    def test_extracts_various_internal_links(self):
        sample = """
            <html>
                <a href="/about">About</a>
                <script src="/js/app.js"></script>
                <img src="https://example.com/img.png" />
                <form action="/submit"></form>
                <div data-url="/dynamic/data"></div>
                <a href="https://example.com/contact">Contact</a>
                <a href="https://other.com/page">External</a>
            </html>
        """
        expected = [
            "https://example.com/about",
            "https://example.com/contact",
            "https://example.com/dynamic/data",
            "https://example.com/img.png",
            "https://example.com/js/app.js",
            "https://example.com/submit",
        ]
        result = extract_internal_links(sample, "https://example.com")
        self.assertEqual(sorted(result), sorted(expected))


class TestClickableRenderedExtraction(unittest.TestCase):
    def test_clickable_text_extraction_html(self):
        html_content = """
        <a href="/home">Link Text</a>
        <button onclick="alert('hi')">Click Me</button>
        <input type="submit" value="Submit">
        <div id="js-btn">JS Button</div>
        <script>
          document.getElementById('js-btn').addEventListener('click', () => console.log('Clicked!'));
        </script>
        """
        result = extract_clickable_texts_from_rendered_page(
            source=html_content)
        # expected = ["Link Text", "Click Me", "Submit", "JS Button"]
        expected = ["Link Text", "Click Me", "Submit"]
        self.assertTrue(all(item in result for item in expected))

    def test_clickable_text_extraction_url(self):
        html_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/scrapers/mock/sample_clickable.html"
        result = extract_clickable_texts_from_rendered_page(source=html_path)
        # expected = ["Link Text", "Click Me", "Submit", "Another Button", "JS Button"]
        expected = ["Link Text", "Click Me", "Submit", "Another Button"]
        self.assertTrue(all(item in result for item in expected))


class TestElementScreenshotter(unittest.TestCase):
    def test_screenshot_html(self):
        html_content = """
        <div class="card" style="width:100px;height:100px;background-color: red;">Card 1</div>
        <div class="card" style="width:100px;height:100px;background-color: blue;">Card 2</div>
        """
        result = extract_element_screenshots(
            source=html_content, css_selectors=[".card"])
        self.assertTrue(all(os.path.exists(path) for path in result))
        self.assertGreater(len(result), 0)

    def test_screenshot_url(self):
        html_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/scrapers/mock/sample_clickable.html"
        result = extract_element_screenshots(
            source=html_path, css_selectors=[".card"])
        self.assertTrue(all(os.path.exists(path) for path in result))
        self.assertGreater(len(result), 0)


class TestExtractFormElements(unittest.TestCase):
    def test_extract_form_elements_with_html(self):
        html_sample = """
        <input id='search_input' class='user-search'>
        <button id='submit_btn' class='btn-primary'>Submit</button>
        <form id='login_form'>
            <input type="text" id="username" class="input-text">
            <textarea id="comments" class="textarea"></textarea>
        </form>
        """
        result = extract_form_elements(html_sample)
        expected = ['#search_input', '.user-search', '#submit_btn', '.btn-primary',
                    '#login_form', '#username', '.input-text', '#comments', '.textarea']
        self.assertEqual(result, expected)

    def test_extract_form_elements_with_url(self):
        # Replace 'url' with an actual URL for a test case
        url = "https://aniwatchtv.to"
        result = extract_form_elements(url)
        self.assertIsInstance(result, list)  # Ensure it returns a list
        # Check that it doesn't return an empty list
        self.assertGreater(len(result), 0)

    def test_extract_form_elements_with_empty_html(self):
        html_sample = ""
        result = extract_form_elements(html_sample)
        self.assertEqual(result, [])

    def test_extract_form_elements_with_no_matching_elements(self):
        html_sample = "<div class='no-form-elements'></div>"
        result = extract_form_elements(html_sample)
        self.assertEqual(result, [])

    def test_extract_form_elements_with_invalid_html(self):
        html_sample = "<input id='valid_input' class='css-valid'>"
        result = extract_form_elements(html_sample)
        self.assertEqual(result, ['#valid_input'])


class TestExtractSearchInputs(unittest.TestCase):
    def test_extract_search_inputs_with_html(self):
        html_sample = """
        <input type='search' id='search_field' class='search-box'>
        <input type='text' id='text_input' class='text-box'>
        """
        result = extract_search_inputs(html_sample)
        expected = ['#search_field', '.search-box', '#text_input', '.text-box']
        self.assertEqual(result, expected)

    def test_extract_search_inputs_with_url(self):
        # Replace 'url' with an actual URL for a test case
        url = "https://aniwatchtv.to"
        result = extract_search_inputs(url)
        self.assertIsInstance(result, list)  # Ensure it returns a list
        # Check that it doesn't return an empty list
        self.assertGreater(len(result), 0)

    def test_extract_search_inputs_with_no_search_inputs(self):
        html_sample = "<div>No search inputs here</div>"
        result = extract_search_inputs(html_sample)
        self.assertEqual(result, [])

    def test_extract_search_inputs_with_invalid_html(self):
        html_sample = "<input type='text' id='valid_input' class='text-box'>"
        result = extract_search_inputs(html_sample)
        self.assertEqual(result, ['#valid_input', '.text-box'])


class TestExtractByHeadingHierarchy(unittest.TestCase):

    def test_single_heading(self):
        html = """
        <html>
            <body>
                <h1>Main Heading</h1>
                <p>Content under main heading</p>
            </body>
        </html>
        """
        result = extract_by_heading_hierarchy(html)

        # Expecting 2 nodes, one for h1 and one for the paragraph under it
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['tag'], 'h1')
        self.assertEqual(result[1]['tag'], 'p')
        self.assertEqual(result[1]['text'], 'Content under main heading')

        # Check depth and parent-child relationships
        self.assertEqual(result[0]['depth'], 0)  # h1 has depth 0
        # p is a child of h1, so it has depth 1
        self.assertEqual(result[1]['depth'], 1)
        self.assertEqual(result[1]['parent'], result[0]
                         ['id'])  # p's parent should be h1

    def test_multiple_headings(self):
        html = """
        <html>
            <body>
                <h1>Main Heading</h1>
                <p>Content under main heading</p>
                <h2>Subheading 1</h2>
                <p>Content under subheading 1</p>
                <h2>Subheading 2</h2>
                <p>Content under subheading 2</p>
            </body>
        </html>
        """
        result = extract_by_heading_hierarchy(html)

        # Expecting 5 nodes (1 h1, 2 h2, and 2 paragraphs under h2)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0]['tag'], 'h1')
        self.assertEqual(result[1]['tag'], 'p')
        self.assertEqual(result[2]['tag'], 'h2')
        self.assertEqual(result[3]['tag'], 'p')
        self.assertEqual(result[4]['tag'], 'h2')

        # Check depth and parent-child relationships
        self.assertEqual(result[0]['depth'], 0)  # h1 has depth 0
        # p is a child of h1, so it has depth 1
        self.assertEqual(result[1]['depth'], 1)
        self.assertEqual(result[1]['parent'], result[0]
                         ['id'])  # p's parent should be h1

        self.assertEqual(result[2]['depth'], 1)  # h2 has depth 1
        # p is a child of h2, so p's depth is 2
        self.assertEqual(result[3]['depth'], 2)
        self.assertEqual(result[3]['parent'], result[2]
                         ['id'])  # p's parent should be h2

        self.assertEqual(result[4]['depth'], 1)  # h2 has depth 1
        # p is a child of h2, so p's depth is 2
        self.assertEqual(result[5]['depth'], 2)
        self.assertEqual(result[5]['parent'], result[4]
                         ['id'])  # p's parent should be h2

    def test_heading_without_content(self):
        html = """
        <html>
            <body>
                <h1>Main Heading</h1>
                <h2>Subheading 1</h2>
                <h3>Sub-subheading</h3>
            </body>
        </html>
        """
        result = extract_by_heading_hierarchy(html)

        # Expecting 4 nodes: h1, h2, h3, and a text node under h2
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0]['tag'], 'h1')
        self.assertEqual(result[1]['tag'], 'h2')
        self.assertEqual(result[2]['tag'], 'h3')

        # Check depth and parent-child relationships
        self.assertEqual(result[0]['depth'], 0)  # h1 has depth 0
        self.assertEqual(result[1]['depth'], 1)  # h2 has depth 1
        self.assertEqual(result[2]['depth'], 2)  # h3 has depth 2

        # No text nodes, so no children should be added to h1, h2, or h3
        self.assertEqual(result[0]['children'], [])
        self.assertEqual(result[1]['children'], [])
        self.assertEqual(result[2]['children'], [])

    def test_no_headings(self):
        html = """
        <html>
            <body>
                <p>Only some content here.</p>
                <p>More content here.</p>
            </body>
        </html>
        """
        result = extract_by_heading_hierarchy(html)

        # Expecting 2 nodes, both paragraphs
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['tag'], 'p')
        self.assertEqual(result[1]['tag'], 'p')

        # Check depth and parent-child relationships
        self.assertEqual(result[0]['depth'], 0)  # p has depth 0
        # p has depth 0, as no headings exist
        self.assertEqual(result[1]['depth'], 0)

    def test_empty_html(self):
        html = """
        <html>
            <body>
            </body>
        </html>
        """
        result = extract_by_heading_hierarchy(html)

        # Expecting an empty list, since there are no headings or content
        self.assertEqual(result, [])

    def test_content_between_headings(self):
        html = """
        <html>
            <body>
                <h1>Main Heading</h1>
                <p>Content under main heading</p>
                <h2>Subheading 1</h2>
                <p>Content under subheading 1</p>
                <h2>Subheading 2</h2>
                <p>Content under subheading 2</p>
                <h3>Sub-subheading</h3>
                <p>Content under sub-subheading</p>
            </body>
        </html>
        """
        result = extract_by_heading_hierarchy(html)

        # Expecting 6 nodes: h1, h2, h2, h3, and paragraphs under each heading
        self.assertEqual(len(result), 6)

        # Check the headings and their corresponding content
        self.assertEqual(result[0]['tag'], 'h1')
        self.assertEqual(result[1]['tag'], 'p')
        self.assertEqual(result[2]['tag'], 'h2')
        self.assertEqual(result[3]['tag'], 'p')
        self.assertEqual(result[4]['tag'], 'h2')
        self.assertEqual(result[5]['tag'], 'p')

        # Check depth and parent-child relationships
        self.assertEqual(result[0]['depth'], 0)  # h1 has depth 0
        # p is a child of h1, so it has depth 1
        self.assertEqual(result[1]['depth'], 1)
        self.assertEqual(result[1]['parent'], result[0]
                         ['id'])  # p's parent should be h1

        self.assertEqual(result[2]['depth'], 1)  # h2 has depth 1
        # p is a child of h2, so p's depth is 2
        self.assertEqual(result[3]['depth'], 2)
        self.assertEqual(result[3]['parent'], result[2]
                         ['id'])  # p's parent should be h2

        self.assertEqual(result[4]['depth'], 1)  # h2 has depth 1
        # p is a child of h2, so p's depth is 2
        self.assertEqual(result[5]['depth'], 2)
        self.assertEqual(result[5]['parent'], result[4]
                         ['id'])  # p's parent should be h2

    def test_custom_tags_to_split_on(self):
        html = """
        <html>
            <body>
                <h1>Main Heading</h1>
                <p>Content under main heading</p>
                <h2>Subheading 1</h2>
                <p>Content under subheading 1</p>
            </body>
        </html>
        """
        custom_tags = ['h1', 'h3']
        result = extract_by_heading_hierarchy(
            html, tags_to_split_on=custom_tags)

        # We should have only the h1 and h3 as root nodes, with text in between
        # 2 nodes for h1 and h3, plus 2 paragraphs
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0]['tag'], 'h1')
        self.assertEqual(result[2]['tag'], 'h3')
        self.assertEqual(result[1]['tag'], 'p')
        self.assertEqual(result[3]['tag'], 'p')

        # Check depth and parent-child relationships
        self.assertEqual(result[0]['depth'], 0)  # h1 has depth 0
        # p is a child of h1, so p's depth is 1
        self.assertEqual(result[1]['depth'], 1)
        self.assertEqual(result[1]['parent'], result[0]
                         ['id'])  # p's parent should be h1

        # h3 has depth 0 because of custom tags
        self.assertEqual(result[2]['depth'], 0)
        # p is a child of h3, so p's depth is 1
        self.assertEqual(result[3]['depth'], 1)
        self.assertEqual(result[3]['parent'], result[2]
                         ['id'])  # p's parent should be h3

    def test_heading_with_non_text_children(self):
        html = """
        <html>
            <body>
                <h1>Main Heading</h1>
                <div>Some non-text content</div>
                <p>Content under main heading</p>
            </body>
        </html>
        """
        result = extract_by_heading_hierarchy(html)

        # Expecting 3 nodes: h1, div (ignored), and the paragraph
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['tag'], 'h1')
        self.assertEqual(result[1]['tag'], 'p')
        self.assertEqual(result[2]['tag'], 'p')

        # Check depth and parent-child relationships
        self.assertEqual(result[0]['depth'], 0)  # h1 has depth 0
        # p is a child of h1, so it has depth 1
        self.assertEqual(result[1]['depth'], 1)
        self.assertEqual(result[1]['parent'], result[0]
                         ['id'])  # p's parent should be h1


if __name__ == "__main__":
    unittest.main()
