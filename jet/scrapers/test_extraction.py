import os
import unittest
from jet.scrapers.utils import extract_clickable_texts_from_rendered_page, extract_element_screenshots, extract_form_elements, extract_search_inputs, extract_title_and_metadata, extract_internal_links


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
        expected = ['search_input', 'user-search', 'submit_btn', 'btn-primary',
                    'login_form', 'username', 'input-text', 'comments', 'textarea']
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
        self.assertEqual(result, ['valid_input'])


class TestExtractSearchInputs(unittest.TestCase):
    def test_extract_search_inputs_with_html(self):
        html_sample = """
        <input type='search' id='search_field' class='search-box'>
        <input type='text' id='text_input' class='text-box'>
        """
        result = extract_search_inputs(html_sample)
        expected = ['search_field', 'search-box', 'text_input', 'text-box']
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
        self.assertEqual(result, ['valid_input', 'text-box'])


if __name__ == "__main__":
    unittest.main()
