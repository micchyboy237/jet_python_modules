import pytest
import inflect


@pytest.fixture
def p():
    """Fixture to provide a fresh inflect.engine instance."""
    return inflect.engine()


class TestComparisons:
    def test_compare_nouns_singular_plural(self, p):
        # Given a singular and plural noun
        word1 = "person"
        word2 = "people"
        expected = "s:p"

        # When compare_nouns is called
        result = p.compare_nouns(word1, word2)

        # Then it returns the correct comparison result
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_compare_verbs_identical(self, p):
        # Given two identical verbs
        word1 = "run"
        word2 = "run"
        expected = "eq"

        # When compare_verbs is called
        result = p.compare_verbs(word1, word2)

        # Then it returns 'eq'
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_compare_verbs_different(self, p):
        # Given two different verb forms
        word1 = "run"
        word2 = "ran"
        expected = False

        # When compare_verbs is called
        result = p.compare_verbs(word1, word2)

        # Then it returns False
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_compare_adjs_singular_plural(self, p):
        # Given a singular and plural adjective
        word1 = "my"
        word2 = "our"
        expected = "s:p"

        # When compare_adjs is called
        result = p.compare_adjs(word1, word2)

        # Then it returns the correct comparison result
        assert result == expected, f"Expected '{expected}', but got '{result}'"
