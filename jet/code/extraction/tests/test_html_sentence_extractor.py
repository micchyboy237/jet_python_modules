from jet.code.extraction.html_sentence_extractor import (
    extract_main_text,
    html_to_sentences,
    split_into_sentences,
)

HTML_SAMPLE = """
<html>
<head>
    <title>Example Page</title>
    <style>.hidden{display:none}</style>
    <script>console.log("noise")</script>
</head>
<body>
<nav>Home | About | Contact</nav>
<article>
    <h1>Main Title</h1>
    <p>This is the first sentence.</p>
    <p>This is the second sentence! Is this the third?</p>
</article>
<footer>Copyright 2026</footer>
</body>
</html>
"""


def test_extract_main_text_removes_boilerplate():
    # Given
    html = HTML_SAMPLE

    # When
    result = extract_main_text(html)

    # Then
    assert "Main Title" in result
    assert "This is the first sentence." in result
    assert "console.log" not in result
    assert "Home | About | Contact" not in result


def test_split_into_sentences_precise():
    # Given
    text = "Hello world! This works. OK?"

    # When
    result = split_into_sentences(text)

    # Then
    expected = [
        "Hello world!",
        "This works.",
        "OK?",
    ]
    assert result == expected


def test_html_to_sentences_end_to_end():
    # Given
    html = HTML_SAMPLE

    # When
    result = html_to_sentences(html)

    # Then
    expected = [
        "Main Title",
        "This is the first sentence.",
        "This is the second sentence!",
        "Is this the third?",
    ]
    assert result == expected
