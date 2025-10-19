from jet.code.html_utils import preprocess_html
from jet.transformers.formatters import format_html

class TestPreprocessHTML:
    def test_removes_unwanted_elements(self):
        # Given: HTML with unwanted elements
        html_input = """
        <html>
        <head><title>Test Title</title></head>
        <body>
        <button>Click</button>
        <script>alert('test');</script>
        <style>.test {}</style>
        <form><input type="text"></form>
        <p>Test paragraph</p>
        </body>
        </html>
        """
        expected = """
        <html>
        <head><title>Test Title</title></head>
        <body><h1>Test Title</h1>
        <p>Test paragraph</p>
        </body>
        </html>
        """
        # When: preprocess_html is called
        result = preprocess_html(html_input)
        # Then: Unwanted elements are removed, title is inserted as h1
        assert format_html(result) == format_html(expected)

    def test_converts_dt_dd_to_paragraph(self):
        # Given: HTML with dl containing multiple dt/dd pairs within inner divs
        html_input = """
        <html>
        <head><title>Test Title</title></head>
        <body>
        <dl>
            <div>
                <dt> Term 1 </dt>
                <dd> Definition 1 </dd>
            </div>
            <div>
                <dt> Term 2 </dt>
                <dd> Definition 2 </dd>
            </div>
        </dl>
        </body>
        </html>
        """
        expected = """
        <html>
        <head><title>Test Title</title></head>
        <body><h1>Test Title</h1>
        <ul>
            <li>Definition 1: Term 1</li>
            <li>Definition 2: Term 2</li>
        </ul>
        </body>
        </html>
        """
        # When: preprocess_html is called
        result = preprocess_html(html_input)
        # Then: dt/dd pairs are converted to paragraphs, dl is preserved, inner divs are removed
        assert format_html(result) == format_html(expected)

    def test_adds_space_between_inline_elements(self):
        # Given: HTML with consecutive inline elements
        html_input = """
        <html>
        <head><title>Test Title</title></head>
        <body>
        <span>Test1</span><span>Test2</span>
        </body>
        </html>
        """
        expected = """
        <html>
        <head><title>Test Title</title></head>
        <body><h1>Test Title</h1>
        <span>Test1</span> <span>Test2</span>
        </body>
        </html>
        """
        # When: preprocess_html is called
        result = preprocess_html(html_input)
        # Then: Space is added between inline elements
        assert format_html(result) == format_html(expected)

    def test_preserves_existing_h1(self):
        # Given: HTML with an existing h1 as the first child of body
        html_input = """
        <html>
        <head><title>Test Title</title></head>
        <body>
        <h1>Existing Header</h1>
        <p>Test paragraph</p>
        </body>
        </html>
        """
        expected = """
        <html>
        <head><title>Test Title</title></head>
        <body>
        <h1>Existing Header</h1>
        <p>Test paragraph</p>
        </body>
        </html>
        """
        # When: preprocess_html is called
        result = preprocess_html(html_input)
        # Then: Existing h1 is preserved, no title insertion
        assert format_html(result) == format_html(expected)

    def test_removes_comments(self):
        # Given: HTML with comments
        html_input = """
        <html>
        <head><title>Test Title</title></head>
        <body>
        <!-- This is a comment -->
        <p>Test paragraph</p>
        </body>
        </html>
        """
        expected = """
        <html>
        <head><title>Test Title</title></head>
        <body><h1>Test Title</h1>
        <p>Test paragraph</p>
        </body>
        </html>
        """
        # When: preprocess_html is called
        result = preprocess_html(html_input)
        # Then: Comments are removed
        assert format_html(result) == format_html(expected)

    def test_includes_specific_tags(self):
        # Given: HTML with various tags, including only nav and p
        html_input = """
        <html>
        <head><title>Test Title</title></head>
        <body>
        <nav>Navigation</nav>
        <p>Paragraph</p>
        <footer>Footer</footer>
        <script>alert('test');</script>
        </body>
        </html>
        """
        expected = """
        <html>
        <head><title>Test Title</title></head>
        <body><h1>Test Title</h1>
        <nav>Navigation</nav>
        <p>Paragraph</p>
        </body>
        </html>
        """
        # When: preprocess_html is called with includes=['nav', 'p']
        result = preprocess_html(html_input, includes=['nav', 'p'])
        # Then: Only nav and p tags are kept
        assert format_html(result) == format_html(expected)

    def test_excludes_specific_tags(self):
        # Given: HTML with various tags, excluding nav and footer
        html_input = """
        <html>
        <head><title>Test Title</title></head>
        <body>
        <nav>Navigation</nav>
        <p>Paragraph</p>
        <footer>Footer</footer>
        <div>Content</div>
        </body>
        </html>
        """
        expected = """
        <html>
        <head><title>Test Title</title></head>
        <body><h1>Test Title</h1>
        <p>Paragraph</p>
        <div>Content</div>
        </body>
        </html>
        """
        # When: preprocess_html is called with excludes=['nav', 'footer']
        result = preprocess_html(html_input, excludes=['nav', 'footer'])
        # Then: nav and footer tags are removed
        assert format_html(result) == format_html(expected)
