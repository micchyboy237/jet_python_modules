import multiprocessing
import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from tqdm import tqdm
from jet.wordnet.pos_ner_tagger import POSTaggerProperNouns

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('maxent_ne_chunker_tab')
# nltk.download('words')


class SubjectExtractor:
    def extract_subject(self, text):
        """
        Extract the subject from the text.

        Args:
        text (str): The text to analyze.

        Returns:
        list: A list of subjects found in the text.
        """
        subjects = []

        # Split the text into sentences
        sentences = sent_tokenize(text)

        for sentence in sentences:
            # Tokenize each sentence into words and tag them
            words = word_tokenize(sentence)
            pos_tags = pos_tag(words)

            # Chunking for named entity recognition
            ne_tree = ne_chunk(pos_tags)

            # Loop through each chunk and extract the subject
            for chunk in ne_tree:
                if hasattr(chunk, 'label') and chunk.label() in ['PERSON', 'ORGANIZATION', 'GPE', 'LOCATION']:
                    subjects.append({
                        'text': ' '.join(c[0] for c in chunk),
                        'label': chunk.label()
                    })
                else:
                    # If the word is a proper noun (NNP) or a pronoun (PRP), consider it as a subject
                    if isinstance(chunk, tuple) and chunk[1] in ['NNP', 'PRP']:
                        subjects.append({
                            'text': chunk[0],
                            'label': chunk[1]
                        })

        return subjects


def get_label(label):
    if label == 'PER':
        return 'Person'
    elif label == 'ORG':
        return 'Organization'
    elif label == 'LOC':
        return 'Location'

    return 'Unknown'


# Unit Test
def test_extract_subject():
    extractor = SubjectExtractor()

    text = "Albert Einstein developed the theory of relativity. He was born in Germany."
    actual = extractor.extract_subject(text)
    expected = ['Albert Einstein', 'He', 'Germany']
    assert actual == expected, f"\nExpected: {expected}\nActual: {actual}"


def analyze_subject(text, extractor, model):
    subjects = extractor.extract_subject(text)
    subject = subjects[0]['text']
    pos_ner_results = model.predict(text)

    ner_item = {}

    for item in pos_ner_results:
        matched_pos_words = re.findall(rf'\b{subject}\b', item['span'])
        if matched_pos_words:
            obj = {
                'text': subject,
                'pos': 'PROPN',
                'score': item['score'],
                'label': get_label(item['label'])
            }
            ner_item = obj
            break

    return ner_item


def analyze_subjects(texts):
    extractor = SubjectExtractor()
    model = POSTaggerProperNouns()

    results = []

    for text in texts:
        result = analyze_subject(text, extractor, model)

        if result:
            results.append(result)

    return results


if __name__ == '__main__':
    texts = [
        'Manila, officially the City of Manila, is the capital and second-most populous city of the Philippines.'
    ]
    results = analyze_subjects(texts)

    print(f"Results:\n{results}")

    # Explicitly clean up multiprocessing resources
    multiprocessing.active_children()
