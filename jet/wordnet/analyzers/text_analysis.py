from typing import Dict, Literal, TypedDict, List, Optional
from collections import Counter
from textstat import textstat as ts


OverallDifficultyCategoryType = Literal["very_easy",
                                        "easy", "moderate", "difficult", "very_difficult"]
ScoresCategoryType = Literal["very_low", "low", "medium", "high"]


class TextStats(TypedDict):
    character_count_with_spaces: int
    character_count_without_spaces: int
    letter_count_with_spaces: int
    letter_count_without_spaces: int
    text_without_punctuation: str
    lexicon_count: int
    miniword_count: int
    sentence_count: int
    syllable_count: int
    polysyllable_count: int
    monosyllable_count: int
    avg_sentence_length: float
    avg_syllables_per_word: float
    avg_character_per_word: float
    avg_letter_per_word: float
    avg_sentence_per_word: float
    words_per_sentence: float
    complex_arabic_words: int
    arabic_syllable_count: int
    faseeh_count: int
    arabic_long_words: int
    osman_score: float
    flesch_reading_ease: float
    flesch_reading_ease_category: str
    flesch_kincaid_grade: float
    flesch_kincaid_grade_category: str
    smog_index: float
    smog_index_category: str
    coleman_liau_index: float
    automated_readability_index: float
    automated_readability_index_category: str
    linsear_write_formula: float
    dale_chall_score: float
    dale_chall_score_v2: float
    gunning_fog: float
    gunning_fog_category: str
    lix_score: float
    rix_score: float
    spache_readability: float
    fernandez_huerta: float
    szigriszt_pazos: float
    gutierrez_polini: float
    crawford_score: float
    gulpease_index: float
    wiener_sachtextformel: float
    mcalpine_eflaw: float
    difficult_words_count: int
    difficult_words_list: List[str]
    is_comprehension_difficult: bool
    is_easy_word: bool
    long_word_count: int
    reading_time: float
    text_standard: float
    text_standard_description: str
    mltd: float
    mltd_category: ScoresCategoryType
    overall_difficulty: float
    overall_difficulty_category: OverallDifficultyCategoryType


class MLTDScores(TypedDict):
    text_without_punctuation: str
    lexicon_count: int


class ReadabilityScores(TypedDict):
    flesch_kincaid_grade: float
    flesch_reading_ease: float
    gunning_fog: float
    smog_index: float
    automated_readability_index: float


class ReadabilityResult(TypedDict):
    scores: ReadabilityScores
    categories: Dict[str, str]
    overall_difficulty: float
    overall_difficulty_category: OverallDifficultyCategoryType
    overall_difficulty_description: str


def categorize_score(metric: str, value: float, thresholds) -> str:
    """Categorizes a readability score based on predefined thresholds."""
    drifted_value = value if metric != 'flesch_reading_ease' or value >= 0 else 0
    if metric not in thresholds:
        return "N/A"
    t = thresholds[metric]

    if metric == 'flesch_reading_ease':  # Inverted scale: higher is easier
        if drifted_value > t['very_easy']:
            return "very_easy"
        elif drifted_value > t['easy']:
            return "easy"
        elif drifted_value > t['moderate']:
            return "moderate"
        elif drifted_value > t['difficult']:
            return "difficult"
        else:
            return "very_difficult"
    else:  # Normal scale: higher is harder
        if drifted_value < t['very_easy']:
            return "very_easy"
        elif drifted_value < t['easy']:
            return "easy"
        elif drifted_value < t['moderate']:
            return "moderate"
        elif drifted_value < t['difficult']:
            return "difficult"
        else:
            return "very_difficult"


def get_readability_description(category: OverallDifficultyCategoryType) -> str:
    """Maps full category name to simplified readability value."""
    category_map = {
        "very_easy": "Very Easy (Elementary)",
        "easy": "Easy (Middle School)",
        "moderate": "Moderate (High School)",
        "difficult": "Difficult (College)",
        "very_difficult": "Very Difficult (Specialist)",
    }
    return category_map.get(category, "N/A")


def calculate_mtld(text_stats: MLTDScores) -> float:
    """
    Calculates the Measure of Textual Lexical Diversity (MTLD) for the given text stats.

    Args:
        text_stats: Dictionary containing text analysis metrics from analyze_text,
                   including 'text_without_punctuation' and 'lexicon_count'.

    Returns:
        float: The MTLD score, representing lexical diversity.
               Returns 0.0 for texts with fewer than 10 words or invalid input.
    """
    # Extract required metrics from text_stats
    text = text_stats.get('text_without_punctuation', '')
    lexicon_count = text_stats.get('lexicon_count', 0)

    # Return 0.0 for short or invalid texts
    if not text or lexicon_count < 10:
        return 0.0

    # Tokenize the text into words
    words = ts.remove_punctuation(text).lower().split()

    # Validate lexicon_count against tokenized words
    if len(words) != lexicon_count:
        lexicon_count = len(words)

    # Return 0.0 if still too short
    if lexicon_count < 10:
        return 0.0

    # Define TTR threshold for segmenting
    TTR_THRESHOLD = 0.72

    def compute_factors(word_list: list[str]) -> float:
        """Computes the number of factors and their average length for a word list."""
        factor_count = 0.0  # Use float for partial factors
        current_words = []
        unique_words = set()

        for word in word_list:
            current_words.append(word)
            unique_words.add(word)
            ttr = len(unique_words) / \
                len(current_words) if current_words else 1.0

            # Complete factor only if TTR drops below threshold and segment is long enough
            if ttr < TTR_THRESHOLD and len(current_words) >= 15:
                factor_count += 1
                current_words = []
                unique_words = set()

        # Handle remaining words
        if current_words:
            ttr = len(unique_words) / \
                len(current_words) if current_words else 1.0
            if ttr >= TTR_THRESHOLD or len(current_words) >= 15:
                factor_count += 1  # Count as full factor if diverse or long
            else:
                # Proportional contribution
                factor_count += min(ttr / TTR_THRESHOLD, 1.0)

        # Return average factor length
        total_words = len(word_list)
        return total_words / factor_count if factor_count > 0 else total_words

    # Calculate MTLD in forward and backward directions
    forward_mtld = compute_factors(words)
    backward_mtld = compute_factors(words[::-1])

    # Average the forward and backward MTLD scores
    return (forward_mtld + backward_mtld) / 2


def calculate_mtld_category(mtld_score: float, medium_threshold: float = 60.0) -> ScoresCategoryType:
    """
    Categorizes an MTLD score into one of four lexical diversity categories based on a medium threshold.

    Args:
        mtld_score: The MTLD score calculated by calculate_mtld.
        medium_threshold: The lower bound for the medium category (default 60.0).

    Returns:
        ScoresCategoryType: The category of lexical diversity.
    """
    very_low_threshold = medium_threshold * \
        0.67  # e.g., 40 if medium_threshold = 60
    low_threshold = medium_threshold
    high_threshold = medium_threshold * 1.33  # e.g., 80 if medium_threshold = 60

    if mtld_score < very_low_threshold:
        return "very_low"
    elif mtld_score < low_threshold:
        return "low"
    elif mtld_score < high_threshold:
        return "medium"
    else:
        return "high"


def calculate_overall_difficulty_category(difficulty: float) -> OverallDifficultyCategoryType:
    """Converts a float difficulty score to a categorical label."""
    if difficulty <= 1.5:
        return "very_easy"
    elif difficulty <= 2.5:
        return "easy"
    elif difficulty <= 3.5:
        return "moderate"
    elif difficulty <= 4.5:
        return "difficult"
    else:
        return "very_difficult"


def calculate_overall_difficulty(scores: ReadabilityScores, thresholds: Dict[str, Dict[str, float]]) -> float:
    """Calculates the overall difficulty as a float based on weighted readability scores."""
    weights = {
        'flesch_kincaid_grade': 0.45,
        'flesch_reading_ease': 0.35,
        'gunning_fog': 0.1,
        'smog_index': 0.05,
        'automated_readability_index': 0.05
    }

    # Map categories to numerical values for float calculation
    category_values = {
        "very_easy": 1.0,
        "easy": 2.0,
        "moderate": 3.0,
        "difficult": 4.0,
        "very_difficult": 5.0
    }

    weighted_sum = 0.0
    total_weight = 0.0
    categories = {metric: categorize_score(
        metric, value, thresholds) for metric, value in scores.items()}
    for metric, category_label in categories.items():
        weight = weights.get(metric, 0)
        if category_label in category_values:
            weighted_sum += category_values[category_label] * weight
            total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 3.0


def analyze_text(text: str, miniword_max_size: int = 3, syllable_threshold: int = 2, ms_per_char: float = 14.69) -> TextStats:
    """
    Analyzes text using textstat and returns a typed dictionary of metrics.

    Args:
        text: Input text to analyze
        miniword_max_size: Maximum length for miniword counting
        syllable_threshold: Syllable threshold for difficult word detection
        ms_per_char: Milliseconds per character for reading time calculation

    Returns:
        TextStats: Dictionary containing various text analysis metrics
    """
    # Define thresholds for readability metrics
    thresholds = {
        'flesch_kincaid_grade': {'very_easy': 4, 'easy': 7, 'moderate': 10, 'difficult': 13},
        'flesch_reading_ease': {'very_easy': 80, 'easy': 60, 'moderate': 40, 'difficult': 20},
        'gunning_fog': {'very_easy': 5, 'easy': 8, 'moderate': 11, 'difficult': 14},
        'smog_index': {'very_easy': 6, 'easy': 9, 'moderate': 11, 'difficult': 13},
        'automated_readability_index': {'very_easy': 4, 'easy': 7, 'moderate': 10, 'difficult': 14}
    }

    # Compute readability scores
    scores: ReadabilityScores = {
        'flesch_kincaid_grade': ts.flesch_kincaid_grade(text),
        'flesch_reading_ease': ts.flesch_reading_ease(text),
        'gunning_fog': ts.gunning_fog(text),
        'smog_index': ts.smog_index(text),
        'automated_readability_index': ts.automated_readability_index(text)
    }

    # Compute mltd scores
    mltd_scores: MLTDScores = {
        "text_without_punctuation": ts.remove_punctuation(text),
        "lexicon_count": ts.lexicon_count(text, removepunct=True),
    }
    mltd = calculate_mtld(mltd_scores)
    mltd_category = calculate_mtld_category(mltd)

    overall_difficulty = calculate_overall_difficulty(scores, thresholds)
    overall_difficulty_category = calculate_overall_difficulty_category(
        overall_difficulty)

    return {
        # Character and Letter Counts
        # Counts total characters, including spaces
        "character_count_with_spaces": ts.char_count(text, ignore_spaces=False),
        # Counts total characters, excluding spaces
        "character_count_without_spaces": ts.char_count(text, ignore_spaces=True),
        # Counts only letters, including spaces
        "letter_count_with_spaces": ts.letter_count(text, ignore_spaces=False),
        # Counts only letters, excluding spaces
        "letter_count_without_spaces": ts.letter_count(text, ignore_spaces=True),

        # Text Processing
        # Removes all punctuation from the text
        "text_without_punctuation": mltd_scores["text_without_punctuation"],

        # Word and Sentence Counts
        # Counts total words, optionally removing punctuation
        "lexicon_count": mltd_scores["lexicon_count"],
        # Counts words with 3 or fewer characters
        "miniword_count": ts.miniword_count(text, max_size=miniword_max_size),
        # Counts total sentences in the text
        "sentence_count": ts.sentence_count(text),

        # Syllable Counts
        # Counts total syllables in the text
        "syllable_count": ts.syllable_count(text),
        # Counts words with multiple syllables
        "polysyllable_count": ts.polysyllabcount(text),
        # Counts words with a single syllable
        "monosyllable_count": ts.monosyllabcount(text),

        # Averages
        # Average number of words per sentence
        "avg_sentence_length": ts.avg_sentence_length(text),
        # Average syllables per word in the text
        "avg_syllables_per_word": ts.avg_syllables_per_word(text),
        # Average characters per word, including punctuation
        "avg_character_per_word": ts.avg_character_per_word(text),
        # Average letters per word, excluding punctuation
        "avg_letter_per_word": ts.avg_letter_per_word(text),
        # Ratio of sentences to words
        "avg_sentence_per_word": ts.avg_sentence_per_word(text),
        # Average number of words per sentence (alternative method)
        "words_per_sentence": ts.words_per_sentence(text),

        # Arabic-specific Metrics
        # Counts complex words in Arabic text
        "complex_arabic_words": ts.count_complex_arabic_words(text),
        # Counts syllables in Arabic text
        "arabic_syllable_count": ts.count_arabic_syllables(text),
        # Counts specific Arabic linguistic features
        "faseeh_count": ts.count_faseeh(text),
        # Counts long words in Arabic text
        "arabic_long_words": ts.count_arabic_long_words(text),
        # Readability score for Arabic text
        "osman_score": ts.osman(text),

        # Readability Scores
        # Measures text readability (higher = easier)
        "flesch_reading_ease": scores['flesch_reading_ease'],
        # Category for Flesch Reading Ease score
        "flesch_reading_ease_category": categorize_score('flesch_reading_ease', scores['flesch_reading_ease'], thresholds),
        # Estimates U.S. grade level needed to understand text
        "flesch_kincaid_grade": scores['flesch_kincaid_grade'],
        # Category for Flesch-Kincaid Grade score
        "flesch_kincaid_grade_category": categorize_score('flesch_kincaid_grade', scores['flesch_kincaid_grade'], thresholds),
        # Estimates years of education needed for comprehension
        "smog_index": scores['smog_index'],
        # Category for SMOG Index score
        "smog_index_category": categorize_score('smog_index', scores['smog_index'], thresholds),
        # Estimates U.S. grade level based on characters
        "coleman_liau_index": ts.coleman_liau_index(text),
        # Estimates U.S. grade level for readability
        "automated_readability_index": scores['automated_readability_index'],
        # Simplified value for Automated Readability Index (very_easy, easy, moderate, difficult, very_difficult)
        "automated_readability_index_category": categorize_score('automated_readability_index', scores['automated_readability_index'], thresholds),
        # Estimates readability for technical texts
        "linsear_write_formula": ts.linsear_write_formula(text),
        # Measures readability using familiar words
        "dale_chall_score": ts.dale_chall_readability_score(text),
        # Alternative Dale-Chall calculation
        "dale_chall_score_v2": ts.dale_chall_readability_score_v2(text),
        # Estimates education level needed for comprehension
        "gunning_fog": scores['gunning_fog'],
        # Category for Gunning Fog score
        "gunning_fog_category": categorize_score('gunning_fog', scores['gunning_fog'], thresholds),
        # Measures readability based on sentence length and long words
        "lix_score": ts.lix(text),
        # Simplified readability index based on long words
        "rix_score": ts.rix(text),
        # Estimates readability for younger readers
        "spache_readability": ts.spache_readability(text),

        # Additional Readability Metrics
        # Spanish readability score adapted for general use
        "fernandez_huerta": ts.fernandez_huerta(text),
        # Readability metric for Spanish texts
        "szigriszt_pazos": ts.szigriszt_pazos(text),
        # Readability metric for Spanish texts
        "gutierrez_polini": ts.gutierrez_polini(text),
        # Readability metric focused on sentence complexity
        "crawford_score": ts.crawford(text),
        # Italian readability metric adapted for general use
        "gulpease_index": ts.gulpease_index(text),
        # German readability metric, variant 1
        "wiener_sachtextformel": ts.wiener_sachtextformel(text, variant=1),
        # Measures ease for English as a foreign language learners
        "mcalpine_eflaw": ts.mcalpine_eflaw(text),

        # Word Difficulty Analysis
        # Counts words with syllables above threshold
        "difficult_words_count": ts.difficult_words(text, syllable_threshold=syllable_threshold),
        # Lists words with syllables above threshold
        "difficult_words_list": ts.difficult_words_list(text, syllable_threshold=syllable_threshold),
        # Checks if 'comprehension' exceeds syllable threshold
        "is_comprehension_difficult": ts.is_difficult_word("comprehension", syllable_threshold=syllable_threshold),
        # Checks if 'easy' is below syllable threshold
        "is_easy_word": ts.is_easy_word("easy", syllable_threshold=syllable_threshold),

        # Additional Metrics
        # Counts words exceeding a length threshold
        "long_word_count": ts.long_word_count(text),
        # Estimates reading time based on character count
        "reading_time": ts.reading_time(text, ms_per_char=ms_per_char),
        # Estimates overall grade level of text
        "text_standard": float(ts.text_standard(text, float_output=True)),
        # Text description of the overall grade level
        "text_standard_description": str(ts.text_standard(text, float_output=False)),

        # Diversity Metrics
        "mltd": mltd,
        "mltd_category": mltd_category,

        # Overall Difficulty
        # Now float
        "overall_difficulty": overall_difficulty,
        "overall_difficulty_category": overall_difficulty_category
    }


def analyze_readability(text: str) -> ReadabilityResult:
    """
    Analyzes the readability of a given text using multiple metrics and categorizes difficulty.

    Args:
        text: Input text to analyze

    Returns:
        ReadabilityResult: Dictionary containing scores, category labels, and overall difficulty
    """
    # Define thresholds for readability metrics
    thresholds = {
        'flesch_kincaid_grade': {'very_easy': 4, 'easy': 7, 'moderate': 10, 'difficult': 13},
        'flesch_reading_ease': {'very_easy': 80, 'easy': 60, 'moderate': 40, 'difficult': 20},
        'gunning_fog': {'very_easy': 5, 'easy': 8, 'moderate': 11, 'difficult': 14},
        'smog_index': {'very_easy': 6, 'easy': 9, 'moderate': 11, 'difficult': 13},
        'automated_readability_index': {'very_easy': 4, 'easy': 7, 'moderate': 10, 'difficult': 14}
    }

    # Weights for metrics in overall difficulty calculation
    weights = {
        'flesch_kincaid_grade': 0.45,
        'flesch_reading_ease': 0.35,
        'gunning_fog': 0.1,
        'smog_index': 0.05,
        'automated_readability_index': 0.05
    }

    try:
        # Compute readability scores
        scores: ReadabilityScores = {
            'flesch_kincaid_grade': ts.flesch_kincaid_grade(text),
            'flesch_reading_ease': ts.flesch_reading_ease(text),
            'gunning_fog': ts.gunning_fog(text),
            'smog_index': ts.smog_index(text),
            'automated_readability_index': ts.automated_readability_index(text)
        }
    except Exception as e:
        raise ValueError(f"Error computing readability measures: {e}")

    # Categorize scores
    categories = {metric: categorize_score(
        metric, value, thresholds) for metric, value in scores.items()}

    # Calculate overall difficulty (weighted majority vote)
    weighted_scores = Counter()
    for metric, value in scores.items():
        category_label = categories[metric]
        weight = weights.get(metric, 0)
        weighted_scores[category_label] += weight

    overall_difficulty = calculate_overall_difficulty(scores, thresholds)
    overall_difficulty_category = calculate_overall_difficulty_category(
        overall_difficulty)

    return {
        'scores': scores,
        'categories': categories,
        'overall_difficulty': overall_difficulty,
        'overall_difficulty_category': overall_difficulty_category,
        # Overall Difficulty
        "overall_difficulty_description": get_readability_description(overall_difficulty_category)
    }
