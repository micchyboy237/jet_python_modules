from typing import TypedDict

import pytest
from jet.search.heuristics.generic_search import (
    GenericSearchEngine,
    search_items,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def documents():
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
def extractor():
    def extract(doc):
        return {
            "title": doc["title"],
            "content": doc["content"],
        }

    return extract


# ============================================================
# Core Engine Behavior (Original + Extended)
# ============================================================


class TestGenericSearchEngineCore:
    def test_and_logic(self, documents, extractor):
        engine = GenericSearchEngine(documents, extractor)

        result = engine.search("python developer", logic="AND")

        expected = 1
        assert len(result) == expected
        assert result[0].item["title"] == "Senior Python Developer"

    def test_or_logic(self, documents, extractor):
        engine = GenericSearchEngine(documents, extractor)

        result = engine.search("python marketing", logic="OR")

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
        engine = GenericSearchEngine(documents, extractor)

        result = engine.search(
            "python",
            filter_fn=lambda d: d["category"] == "engineering",
        )

        titles = sorted([r.item["title"] for r in result])

        expected = sorted(
            [
                "Senior Python Developer",
                "Python Data Engineer",
            ]
        )

        assert titles == expected

    def test_highlight_present(self, documents, extractor):
        engine = GenericSearchEngine(documents, extractor)

        result = engine.search("python")

        highlight = result[0].highlights["title"]

        assert "<mark>python</mark>" in highlight.lower()

    def test_empty_query_returns_empty(self, documents, extractor):
        engine = GenericSearchEngine(documents, extractor)

        result = engine.search("")

        expected = []
        assert result == expected

    def test_no_match_returns_empty(self, documents, extractor):
        engine = GenericSearchEngine(documents, extractor)

        result = engine.search("blockchain")

        expected = []
        assert result == expected

    def test_limit_and_offset(self, documents, extractor):
        engine = GenericSearchEngine(documents, extractor)

        result = engine.search("python", limit=1, offset=1)

        expected = 1
        assert len(result) == expected


# ============================================================
# Optional Extractor (Dynamic Dict Logic)
# ============================================================


class TestOptionalExtractorDict:
    def test_dict_without_extractor(self):
        documents = [
            {
                "title": "Backend Python Engineer",
                "content": "Develop APIs using Python and FastAPI",
                "views": 100,
            },
            {
                "title": "Frontend Developer",
                "content": "React and TypeScript",
                "views": 50,
            },
        ]

        engine = GenericSearchEngine(documents)

        result = engine.search("python")

        expected = "Backend Python Engineer"
        assert result[0].item["title"] == expected

    def test_only_string_fields_indexed(self):
        documents = [
            {
                "title": "Data Scientist",
                "content": "Machine learning and Python",
                "salary": 120000,
                "remote": True,
            }
        ]

        engine = GenericSearchEngine(documents)

        result = engine.search("120000")

        expected = []
        assert result == expected


# ============================================================
# TypedDict Support
# ============================================================


class TestTypedDictSupport:
    def test_typed_dict_without_extractor(self):
        class JobDoc(TypedDict):
            title: str
            content: str
            category: str

        documents: list[JobDoc] = [
            {
                "title": "Python Data Engineer",
                "content": "Build ETL pipelines in Python",
                "category": "engineering",
            },
            {
                "title": "Graphic Designer",
                "content": "Create visual assets",
                "category": "design",
            },
        ]

        engine = GenericSearchEngine(documents)

        result = engine.search("etl")

        expected = "Python Data Engineer"
        assert result[0].item["title"] == expected


# ============================================================
# Non-Dict Without Extractor
# ============================================================


class TestNonDictWithoutExtractor:
    def test_raises_type_error(self):
        class Job:
            def __init__(self, title: str, content: str):
                self.title = title
                self.content = content

        documents = [
            Job("Python Dev", "Work with Python"),
        ]

        with pytest.raises(TypeError):
            GenericSearchEngine(documents)


# ============================================================
# Field Weights
# ============================================================


class TestFieldWeights:
    def test_title_weight_boost(self):
        documents = [
            {
                "title": "Python Architect",
                "content": "General backend engineering",
            },
            {
                "title": "Backend Engineer",
                "content": "Python systems and APIs",
            },
        ]

        engine = GenericSearchEngine(
            documents,
            field_weights={"title": 3.0},
        )

        result = engine.search("python")

        expected = "Python Architect"
        assert result[0].item["title"] == expected


# ============================================================
# Stateless Helper
# ============================================================


class TestStatelessHelper:
    def test_with_extractor(self, documents, extractor):
        result = search_items(
            documents,
            "seo",
            text_extractor=extractor,
        )

        expected = "Marketing Specialist"
        assert result[0].item["title"] == expected

    def test_without_extractor(self):
        documents = [
            {
                "title": "SEO Specialist",
                "content": "Optimize search engines",
            },
            {
                "title": "Backend Developer",
                "content": "Build APIs",
            },
        ]

        result = search_items(
            documents,
            "optimize",
        )

        expected = "SEO Specialist"
        assert result[0].item["title"] == expected


# ============================================================
# AND Edge Case (Single Term)
# ============================================================


class TestSingleTermAndLogic:
    def test_single_term_and(self):
        documents = [
            {"title": "Python Developer", "content": "APIs"},
            {"title": "Java Developer", "content": "Spring"},
        ]

        engine = GenericSearchEngine(documents)

        result = engine.search("python", logic="AND")

        expected = 1
        assert len(result) == expected
