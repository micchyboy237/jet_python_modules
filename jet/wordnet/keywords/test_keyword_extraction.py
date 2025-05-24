# Sample usage
from llama_index.core.indices.keyword_table.utils import extract_keywords_given_response


if __name__ == "__main__":
    text = "I love watching Naruto and Attack on Titan. Have you seen One Piece?"
    anime_titles = extract_keywords_given_response(
        text,
        stop_words=['love', 'watching', 'seen']  # Filter out common words
    )
    # Expected: ['naruto', 'attack on titan', 'one piece', ...]
    print(anime_titles)

# Pytest test


def test_keybert_query_extraction():
    text = "I love watching Naruto and Attack on Titan."
    extracted = extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        top_n=5,  # Increased to ensure 'naruto' and 'attack on titan' are captured
        stop_words=['love', 'watching']
    )
    print(extracted)  # Debug: Print extracted keywords
    assert set(['attack on titan', 'naruto']).issubset(extracted)


if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])
