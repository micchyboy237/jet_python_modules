import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from jet.file.utils import load_data

# nltk.download('stopwords')
# nltk.download('punkt')


class StopWords:
    def __init__(self):
        # English stop words
        self.english_stop_words = list(set(stopwords.words('english')))

        # Tagalog stop words organized by type
        tl_stopwords = load_data(
            "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/wordnet/data/tl_stopwords.json")
        # flatten the list of lists
        tl_stopwords = [word.lower() for sublist in tl_stopwords.values()
                        for word in sublist]

        # Combine all Tagalog stop words
        self.tagalog_stop_words = list(set(tl_stopwords))

    def remove_stop_words(self, text, language='english'):
        if language not in ['english', 'tagalog']:
            raise ValueError(
                "Unsupported language. Choose 'english' or 'tagalog'.")

        # Tokenize the text and remove punctuation
        tokens = word_tokenize(text)
        words = [word for word in tokens if word.isalpha()]

        if language == 'english':
            filtered_text = [
                word for word in words if word.lower() not in self.english_stop_words]
        else:  # tagalog
            filtered_text = [
                word for word in words if word.lower() not in self.tagalog_stop_words]

        return ' '.join(filtered_text)
