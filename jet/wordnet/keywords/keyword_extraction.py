from keybert import KeyBERT
from jet.logger import logger
from jet.transformers.formatters import format_json


def extract_keywords(
    text: str,
    query: str = None,
    model_name: str = "all-mpnet-base-v2",
    keyphrase_ngram_range: tuple = (1, 3),
    top_n: int = 5,
    use_mmr: bool = True,
    diversity: float = 0.5,
    stop_words: list = None
) -> list:
    """
    Extract keywords from text using KeyBERT, optionally guided by a query.

    Args:
        text (str): Input text to extract keywords from.
        query (str, optional): Query to guide keyword extraction (e.g., 'anime').
        model_name (str): Name of the KeyBERT model (default: 'all-MiniLM-L12-v2').
        keyphrase_ngram_range (tuple): Range of n-grams for keyphrases (default: (1, 3)).
        top_n (int): Number of keywords to extract (default: 5).
        use_mmr (bool): Use Maximal Marginal Relevance for diversity (default: True).
        diversity (float): Diversity parameter for MMR (default: 0.5).
        stop_words (list): List of stop words to filter out (default: None).

    Returns:
        list: List of extracted keywords relevant to the query (if provided).
    """
    try:
        # Initialize KeyBERT model
        model = KeyBERT(model_name)

        # If query is provided, append it to the text to bias extraction
        input_text = f"{text} {query}" if query else text

        # Extract keywords
        keywords = model.extract_keywords(
            input_text,
            keyphrase_ngram_range=keyphrase_ngram_range,
            top_n=top_n * 2,  # Extract more candidates to filter later
            use_mmr=use_mmr,
            diversity=diversity,
            stop_words=stop_words
        )

        # Extract keyword strings
        extracted_keywords = [kw[0] for kw in keywords]

        # If query is provided, filter keywords to those related to the query
        if query:
            # Simple heuristic: keep keywords that are not the query itself
            # and are likely related (e.g., contain or are similar to query terms)
            from sentence_transformers import SentenceTransformer, util
            embedding_model = SentenceTransformer(model_name)
            query_embedding = embedding_model.encode(
                query, convert_to_tensor=True)
            filtered_keywords = []
            for keyword in extracted_keywords:
                if keyword.lower() != query.lower():  # Exclude the query itself
                    keyword_embedding = embedding_model.encode(
                        keyword, convert_to_tensor=True)
                    similarity = util.cos_sim(
                        query_embedding, keyword_embedding).item()
                    if similarity > 0.3:  # Threshold for relevance
                        filtered_keywords.append(keyword)
            extracted_keywords = filtered_keywords[:top_n]  # Limit to top_n

        # Log the extracted keywords
        logger.success(format_json(extracted_keywords))
        return extracted_keywords

    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return []


# Pytest test


def test_keybert_query_extraction():
    text = "I love watching Naruto and Attack on Titan."
    extracted = extract_keywords(
        text,
        query="anime",
        keyphrase_ngram_range=(1, 3),
        top_n=5,
        stop_words=['love', 'watching']
    )
    print(extracted)  # Debug: Print extracted keywords
    assert set(['attack on titan', 'naruto']).issubset(extracted)


if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])
