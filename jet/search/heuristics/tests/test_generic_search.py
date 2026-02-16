import pytest
from jet.search.heuristics.generic_search import GenericSearchEngine, search_items


class TestGenericSearchEngine:
    @pytest.fixture
    def documents(self):
        return [
            {
                "title": "Senior Python Developer",
                "content": "Build scalable backend systems using Python",
                "category": "engineering",
            },
            {
                "title": "Marketing Specialist",
                "content": "Digital marketing and SEO campaigns",
                "category": "marketing",
            },
            {
                "title": "Python Data Engineer",
                "content": "Data pipelines and ETL systems",
                "category": "engineering",
            },
        ]

    @pytest.fixture
    def extractor(self):
        def extract(doc):
            return {
                "title": doc["title"],
                "content": doc["content"],
            }

        return extract

    def test_and_logic(self, documents, extractor):
        # Given
        engine = GenericSearchEngine(documents, extractor)

        # When
        result = engine.search("python developer", logic="AND")

        # Then
        expected = 1
        assert len(result) == expected
        assert result[0].item["title"] == "Senior Python Developer"

    def test_or_logic(self, documents, extractor):
        # Given
        engine = GenericSearchEngine(documents, extractor)

        # When
        result = engine.search("python marketing", logic="OR")

        # Then
        titles = sorted([r.item["title"] for r in result])
        expected = sorted(
            [
                "Senior Python Developer",
                "Python Data Engineer",
                "Marketing Specialist",
            ]
        )
        assert titles == expected

    def test_filter_function(self, documents, extractor):
        # Given
        engine = GenericSearchEngine(documents, extractor)

        # When
        result = engine.search(
            "python",
            filter_fn=lambda d: d["category"] == "engineering",
        )

        # Then
        titles = sorted([r.item["title"] for r in result])
        expected = sorted(
            [
                "Senior Python Developer",
                "Python Data Engineer",
            ]
        )
        assert titles == expected

    def test_highlight_present(self, documents, extractor):
        # Given
        engine = GenericSearchEngine(documents, extractor)

        # When
        result = engine.search("python")

        # Then
        highlight = result[0].highlights["title"]
        assert "<mark>python</mark>" in highlight.lower()

    def test_stateless_helper(self, documents, extractor):
        # Given
        query = "seo"

        # When
        result = search_items(
            documents,
            query,
            text_extractor=extractor,
        )

        # Then
        expected = "Marketing Specialist"
        assert result[0].item["title"] == expected
