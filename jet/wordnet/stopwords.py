import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from jet.file.utils import load_data
from typing import List, Set


class StopWords:
    # Static attributes initialized once
    english_stop_words: Set[str] = set(stopwords.words('english'))
    tl_stopwords = load_data(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/wordnet/data/tl_stopwords.json")
    tagalog_stop_words: Set[str] = set(
        word.lower() for sublist in tl_stopwords.values() for word in sublist
    )

    @classmethod
    def remove_stop_words(cls, text: str, language: str = 'english') -> str:
        if language not in ['english', 'tagalog']:
            raise ValueError(
                "Unsupported language. Choose 'english' or 'tagalog'.")

        # Tokenize the text and remove punctuation
        tokens = word_tokenize(text)
        words = [word for word in tokens if word.isalpha()]

        if language == 'english':
            filtered_text = [
                word for word in words if word.lower() not in cls.english_stop_words]
        else:  # tagalog
            filtered_text = [
                word for word in words if word.lower() not in cls.tagalog_stop_words]

        return ' '.join(filtered_text)
