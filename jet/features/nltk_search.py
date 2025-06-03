import nltk
from typing import List, Set, TypedDict
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

ALLOWED_POS: Set[str] = {
    'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
    'PRP', 'PRP$',               # Pronouns
    'JJ', 'JJR', 'JJS',          # Adjectives
    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
    'RB', 'RBR', 'RBS',          # Adverbs
    'CD'                         # Cardinal numbers
}


class PosTag(TypedDict):
    word: str
    pos: str
    lemma: str
    start_idx: int
    end_idx: int


class SearchResult(TypedDict):
    doc_index: int
    matching_words_count: int
    matching_words_with_pos_and_lemma: List[PosTag]
    text: str


def get_wordnet_pos(nltk_pos: str) -> str:
    if nltk_pos.startswith('J'):
        return wordnet.ADJ
    elif nltk_pos.startswith('V'):
        return wordnet.VERB
    elif nltk_pos.startswith('N'):
        return wordnet.NOUN
    elif nltk_pos.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def get_pos_tag(sentence: str) -> List[PosTag]:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Tokenize the sentence and keep track of character indices
    tokens: List[str] = word_tokenize(sentence.lower())
    pos_tags: List[tuple[str, str]] = nltk.pos_tag(tokens)

    # Calculate character indices
    tagged_words: List[PosTag] = []
    current_idx = 0
    sentence_lower = sentence.lower()

    for word, pos in pos_tags:
        # Find the word in the original sentence starting from current_idx
        start_idx = sentence_lower.find(word, current_idx)
        if start_idx == -1:
            continue  # Skip if word not found (rare, but for robustness)
        end_idx = start_idx + len(word)
        if pos in ALLOWED_POS and word not in stop_words:
            tagged_words.append({
                'word': word,
                'pos': pos,
                'lemma': lemmatizer.lemmatize(word, get_wordnet_pos(pos)),
                'start_idx': start_idx,
                'end_idx': end_idx
            })
        current_idx = end_idx

    return tagged_words


def search_by_pos(query: str, documents: List[str]) -> List[SearchResult]:
    query_pos: List[PosTag] = get_pos_tag(query)
    query_lemmas: Set[str] = {pos_tag['lemma'] for pos_tag in query_pos}
    results: List[SearchResult] = []
    for idx, doc in enumerate(documents):
        doc_pos: List[PosTag] = get_pos_tag(doc)
        doc_lemmas: Set[str] = {pos_tag['lemma'] for pos_tag in doc_pos}
        matches: Set[str] = query_lemmas.intersection(doc_lemmas)
        match_count: int = len(matches)
        matching_words_with_pos: List[PosTag] = [
            pos_tag for pos_tag in doc_pos if pos_tag['lemma'] in matches
        ]
        results.append({
            'doc_index': idx,
            'matching_words_count': match_count,
            'matching_words_with_pos_and_lemma': matching_words_with_pos,
            'text': doc
        })
        print(f"Document {idx}: Matches = {matches}, Count = {match_count}")
    results.sort(
        key=lambda x: (-x['matching_words_count'], -x['doc_index'])
    )
    print("Sorted results:", [(r['doc_index'],
          r['matching_words_count']) for r in results])
    return results


if __name__ == "__main__":
    docs: List[str] = [
        "The quick brown fox jumps over the lazy dog",
        "A fox fled from danger",
        "The dog sleeps peacefully",
        "Quick foxes climb steep hills"
    ]
    query: str = "The quick foxes run dangerously"
    results: List[SearchResult] = search_by_pos(query, docs)
    for result in results:
        print(f"Document {result['doc_index']}:")
        print(f"  Text: {result['text']}")
        print(
            f"  Matching words (word, POS, lemma, start_idx, end_idx): "
            f"{[(pos_tag['word'], pos_tag['pos'], pos_tag['lemma'], pos_tag['start_idx'], pos_tag['end_idx']) for pos_tag in result['matching_words_with_pos_and_lemma']]}"
        )
        print(f"  Match count: {result['matching_words_count']}\n")
