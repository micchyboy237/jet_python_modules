from jet.wordnet.similarity import filter_different_texts


class TestFilterDifferentTexts:
    def test_filter_different_texts_identical(self):
        texts = ["Hello world", "Hello world", "Hello world"]
        expected = ["Hello world"]
        result = filter_different_texts(texts)
        assert len(result) == len(
            expected), f"Expected {len(expected)} item, got {len(result)}"
        assert result == expected, f"Expected {expected}, got {result}"

    def test_filter_different_texts_similar(self):
        texts = [
            "This is a sentence.",
            "This is a sentence!",
            "A completely different sentence."
        ]
        expected = [
            "This is a sentence.",
            "A completely different sentence."
        ]
        result = filter_different_texts(texts)
        assert result == expected, f"Expected {expected}, got {result}"

    def test_filter_different_texts_all_different(self):
        texts = ["Hello world", "Goodbye world", "How are you"]
        expected = texts
        result = filter_different_texts(texts)
        assert len(result) == len(
            expected), f"Expected {len(expected)} items, got {len(result)}"
        assert result == expected, f"Expected {expected}, got {result}"

    def test_filter_different_texts_urls(self):
        urls = [
            "http://example.com/page1",
            "https://example.com/page1/",
            "http://example.com/page2",
            "http://different.com"
        ]
        expected = [
            "http://example.com/page1",
            "http://different.com"
        ]
        result = filter_different_texts(urls)
        assert result == expected, f"Expected {expected}, got {result}"

    def test_filter_different_texts_empty(self):
        texts = []
        expected = []
        result = filter_different_texts(texts)
        assert result == expected, f"Expected {expected}, got {result}"
