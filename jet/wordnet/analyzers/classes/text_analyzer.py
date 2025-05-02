import os
import shutil
from jet.file.utils import load_file, save_file
import nltk
import spacy
import numpy as np
from typing import List, Dict, Set, TypedDict
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from spacy.tokens import Doc
from jet.wordnet.sentence import is_sentence, split_sentences
from collections import Counter

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Updated TypedDicts to include POS-related fields


class WordInfo(TypedDict):
    word_count: int
    unique_words: int
    frequency_distribution: Dict[str, int]
    word_complexity_score: float
    word_complexity_category: str
    pos_complexity_score: float
    pos_complexity_category: str
    pos_distribution: Dict[str, int]
    pos_texts: Dict[str, List[str]]  # New: POS tag to list of texts


class SentenceInfo(TypedDict):
    sentence_count: int
    avg_sentence_length: float
    sentences: List[str]
    sentence_complexity_score: float
    sentence_complexity_category: str
    complex_phrases_count: int


class AnalysisResult(TypedDict):
    word_info: WordInfo
    sentence_info: SentenceInfo
    lexical_diversity_mtld: float
    overall_readability_score: float
    readability_category: str


class TextAnalyzer:
    """A class for analyzing text to extract word, sentence, and diversity metrics with evaluation scores."""

    def __init__(self, language: str = 'english') -> None:
        """Initialize with language and load spaCy model."""
        self.language: str = language
        self.stop_words: Set[str] = set(stopwords.words(language))
        try:
            self.nlp: spacy.language.Language = spacy.load(
                'en_core_web_sm' if language == 'english' else 'xx_ent_wiki_sm'
            )
        except OSError:
            raise Exception(
                "Please install spaCy model: python -m spacy download en_core_web_sm")

    def preprocess_text(self, text: str) -> List[str]:
        """Clean text by removing punctuation and converting to lowercase."""
        doc: Doc = self.nlp(text.lower())
        tokens: List[str] = [token.text for token in doc if not token.is_punct]
        return tokens

    def categorize_score(self, score: float, low_threshold: float, high_threshold: float) -> str:
        """Categorize a score as low, medium, or high based on thresholds."""
        if score < low_threshold:
            return "low"
        elif score > high_threshold:
            return "high"
        else:
            return "medium"

    def get_word_info(self, text: str) -> WordInfo:
        """Return word count, unique words, frequency distribution, complexity scores, and POS metrics."""
        tokens: List[str] = self.preprocess_text(text)
        content_tokens: List[str] = [
            t for t in tokens if t not in self.stop_words]

        word_count: int = len(content_tokens)
        unique_words: int = len(set(content_tokens))
        freq_dist: FreqDist = FreqDist(content_tokens)

        # Calculate word complexity (average word length + POS weight)
        doc: Doc = self.nlp(' '.join(content_tokens))
        word_lengths: List[int] = [len(token.text) for token in doc]
        avg_word_length: float = np.mean(word_lengths) if word_lengths else 0.0

        # POS distribution and texts
        pos_tags: List[str] = [
            token.pos_ for token in doc if not token.is_punct]
        pos_dist: Dict[str, int] = dict(Counter(pos_tags))
        pos_texts: Dict[str, List[str]] = {}
        for token in doc:
            if not token.is_punct:
                pos = token.pos_
                if pos not in pos_texts:
                    pos_texts[pos] = []
                pos_texts[pos].append(token.text)

        # Content word ratio (nouns, verbs, adjectives, adverbs)
        content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV'}
        content_word_count: int = sum(pos_dist.get(pos, 0)
                                      for pos in content_pos)
        content_word_ratio: float = content_word_count / \
            len(pos_tags) if pos_tags else 0.0

        # Word complexity: combine avg word length and content word ratio
        word_complexity: float = round(
            (avg_word_length * 0.6 + content_word_ratio * 4.0), 2)
        word_complexity_category: str = self.categorize_score(
            word_complexity, low_threshold=4.0, high_threshold=6.0)

        # POS complexity: measure diversity of POS tags using Shannon entropy
        pos_counts = np.array(list(pos_dist.values()))
        pos_probs = pos_counts / \
            pos_counts.sum() if pos_counts.sum() > 0 else np.array([1.0])
        pos_entropy = -np.sum(pos_probs * np.log2(pos_probs + 1e-10))
        pos_complexity: float = round(pos_entropy, 2)
        pos_complexity_category: str = self.categorize_score(
            pos_complexity, low_threshold=1.5, high_threshold=3.0)

        return {
            'word_count': word_count,
            'unique_words': unique_words,
            'frequency_distribution': dict(freq_dist.most_common()),
            'word_complexity_score': word_complexity,
            'word_complexity_category': word_complexity_category,
            'pos_complexity_score': pos_complexity,
            'pos_complexity_category': pos_complexity_category,
            'pos_distribution': pos_dist,
            'pos_texts': pos_texts
        }

    def get_sentence_info(self, text: str) -> SentenceInfo:
        """Return sentence metrics, complexity scores, and POS-based phrase counts."""
        sentences: List[str] = split_sentences(text)
        valid_sentences: List[str] = [
            sent for sent in sentences if is_sentence(sent)]
        sentence_lengths: List[int] = [
            len(word_tokenize(sent)) for sent in valid_sentences]
        avg_sentence_length: float = np.mean(
            sentence_lengths) if sentence_lengths else 0.0

        # Calculate sentence complexity (length + clauses + complex phrases)
        clause_counts: List[int] = [
            sum(1 for token in self.nlp(sent)
                if token.dep_ in ['ccomp', 'advcl', 'relcl'])
            for sent in valid_sentences
        ]

        # Count complex phrases (e.g., adjective-noun sequences)
        complex_phrases_count: int = 0
        for sent in valid_sentences:
            doc: Doc = self.nlp(sent)
            tokens = [token for token in doc]
            for i in range(len(tokens) - 1):
                if tokens[i].pos_ == 'ADJ' and tokens[i + 1].pos_ == 'NOUN':
                    complex_phrases_count += 1

        # Sentence complexity: combine length, clauses, and phrase count
        sentence_complexity: float = round(
            (avg_sentence_length * 0.5 + np.mean(clause_counts if clause_counts else [0]) * 0.3 +
             complex_phrases_count * 0.2) / 1.5, 2
        )
        sentence_complexity_category: str = self.categorize_score(
            sentence_complexity, low_threshold=10.0, high_threshold=20.0)

        return {
            'sentence_count': len(valid_sentences),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'sentences': valid_sentences,
            'sentence_complexity_score': sentence_complexity,
            'sentence_complexity_category': sentence_complexity_category,
            'complex_phrases_count': complex_phrases_count
        }

    def calculate_readability_score(self, word_info: WordInfo, sentence_info: SentenceInfo, mtld: float) -> float:
        """Calculate readability score incorporating POS-based metrics."""
        word_complexity_norm = min(
            word_info['word_complexity_score'] / 10, 1.0)
        sentence_complexity_norm = min(
            sentence_info['sentence_complexity_score'] / 20, 1.0)
        lexical_diversity_norm = min(mtld / 100, 1.0)
        pos_complexity_norm = min(word_info['pos_complexity_score'] / 5, 1.0)

        readability = (
            0.25 * word_complexity_norm +
            0.35 * sentence_complexity_norm +
            0.25 * lexical_diversity_norm +
            0.15 * pos_complexity_norm
        ) * 100
        return round(readability, 2)

    def analyze(self, text: str) -> AnalysisResult:
        """Perform full text analysis with evaluation scores and categories."""
        word_info = self.get_word_info(text)
        sentence_info = self.get_sentence_info(text)
        lexical_diversity_mtld = self.calculate_mtld(text)
        readability_score = self.calculate_readability_score(
            word_info, sentence_info, lexical_diversity_mtld)
        readability_category = self.categorize_score(
            readability_score, low_threshold=40.0, high_threshold=70.0)

        return {
            'word_info': word_info,
            'sentence_info': sentence_info,
            'lexical_diversity_mtld': lexical_diversity_mtld,
            'overall_readability_score': readability_score,
            'readability_category': readability_category
        }

    def calculate_mtld(self, text: str, min_tokens: int = 10) -> float:
        """Calculate Measure of Textual Lexical Diversity (MTLD)."""
        tokens: List[str] = self.preprocess_text(text)
        content_tokens: List[str] = [
            t for t in tokens if t not in self.stop_words]

        if len(content_tokens) < min_tokens:
            return 0.0

        def ttr_segment(tokens: List[str], threshold: float = 0.72) -> List[List[str]]:
            segments: List[List[str]] = []
            current_tokens: List[str] = []
            for i, token in enumerate(tokens):
                current_tokens.append(token)
                unique: int = len(set(current_tokens))
                total: int = len(current_tokens)
                ttr: float = unique / total if total > 0 else 1.0
                if ttr < threshold or i == len(tokens) - 1:
                    segments.append(current_tokens)
                    current_tokens = []
            return segments

        forward_segments: List[List[str]] = ttr_segment(content_tokens)
        backward_segments: List[List[str]] = ttr_segment(content_tokens[::-1])

        forward_length: float = len(
            content_tokens) / len(forward_segments) if forward_segments else 0.0
        backward_length: float = len(
            content_tokens) / len(backward_segments) if backward_segments else 0.0

        return round((forward_length + backward_length) / 2, 2)


__all__ = [
    "TextAnalyzer",
]

if __name__ == "__main__":
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/searched_html_myanimelist_net_Isekai/merged_docs.json"
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/generated/run_format_html"

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    docs: list[dict] = load_file(data_file)
    docs_with_text_analysis = []
    for doc in docs:
        text = doc["text"]

        # Initialize the TextAnalyzer
        analyzer = TextAnalyzer(language='english')

        # Analyze the text
        analysis = analyzer.analyze(text)

        docs_with_text_analysis.append({
            "text": text,
            "token_count": doc["token_count"],
            "analysis": analysis
        })

        save_file(docs_with_text_analysis,
                  f"{output_dir}/docs_with_text_analysis.json")
