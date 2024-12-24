from difflib import SequenceMatcher, ndiff, get_close_matches


def score_texts_similarity(text1, text2, isjunk=lambda x: not x.isalnum()):
    """Calculate similarity score between two texts using SequenceMatcher."""
    score = SequenceMatcher(isjunk, text1, text2, autojunk=False).ratio()
    return score


def are_texts_similar(text1, text2, threshold=0.7):
    """Check if two texts are similar based on a threshold."""
    return score_texts_similarity(text1, text2) >= threshold


def filter_similar_texts(texts, threshold=0.7):
    """Filter texts that are similar to each other."""
    filtered_texts = []
    for text in texts:
        if any(are_texts_similar(text, existing_text, threshold) for existing_text in filtered_texts):
            continue
        filtered_texts.append(text)
    return filtered_texts


def filter_different_texts(texts, threshold=0.7):
    """Filter texts that are different from each other."""
    filtered_texts = []
    for text in texts:
        if all(not are_texts_similar(text, existing_text, threshold) for existing_text in filtered_texts):
            filtered_texts.append(text)
    return filtered_texts


def get_similar_texts(texts: list[str], threshold: float = 0.7) -> list[dict[str, str]]:
    """Return a list of dictionaries with similar text pairs and their similarity score based on the given threshold."""
    similar_text_pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarity_score = score_texts_similarity(texts[i], texts[j])
            if similarity_score >= threshold:
                similar_text_pairs.append({
                    'text1': texts[i],
                    'text2': texts[j],
                    'score': similarity_score
                })
    return similar_text_pairs


def get_different_texts(texts: list[str], threshold: float = 0.7) -> list[dict[str, str]]:
    """Return a list of dictionaries with different text pairs and their similarity score based on the given threshold."""
    different_text_pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarity_score = score_texts_similarity(texts[i], texts[j])
            if similarity_score < threshold:
                different_text_pairs.append({
                    'text1': texts[i],
                    'text2': texts[j],
                    'score': similarity_score
                })
    return different_text_pairs


def differences(texts, **kwargs):
    """Get differences between adjacent texts."""
    all_differences = []
    for i in range(len(texts) - 1):
        diff = ndiff(texts[i].split(), texts[i + 1].split(), **kwargs)
        differences = [line[2:] for line in diff if line.startswith(
            '+ ') or line.startswith('- ')]
        all_differences.append(
            {'text1': texts[i], 'text2': texts[i + 1], 'differences': differences})
    return all_differences


def similars(texts, **kwargs):
    """Get similarities between adjacent texts."""
    all_similars = []
    for i in range(len(texts) - 1):
        diff = ndiff(texts[i].split(), texts[i + 1].split(), **kwargs)
        similars = [line.strip() for line in diff if not line.startswith(
            '+ ') and not line.startswith('- ')]
        all_similars.append(
            {'text1': texts[i], 'text2': texts[i + 1], 'similars': similars})
    return all_similars


def compare_text_pairs(texts, **kwargs):
    """Compare pairs of texts for similarities and differences."""
    comparisons = []
    for i in range(len(texts) - 1):
        diff = list(ndiff(texts[i].split(), texts[i + 1].split(), **kwargs))
        similarities = [line.strip() for line in diff if line.startswith('  ')]
        differences = [line[2:] for line in diff if line.startswith(
            '+ ') or line.startswith('- ')]
        comparisons.append({
            'text1': texts[i],
            'text2': texts[i + 1],
            'similarities': similarities,
            'differences': differences
        })
    return comparisons


if __name__ == '__main__':
    base_sentence = "October seven is the date of our vacation to Camarines Sur."
    sentences_to_compare = [
        'October 7 ang holiday namin sa Camarines Sur.',
        'October 7 ang araw na nagbakasyon kami sa Camarines Sur.',
        'Ikapito ng Oktubre ang araw ng aming bakasyon sa Camarines Sur.'
    ]

    print(f"Base sentence:\n{base_sentence}")
    result = differences(sentences_to_compare)
    print("Text differences:")
    print(result)
