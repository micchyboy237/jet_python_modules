import multiprocessing
import nltk
import re
import logging
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from tqdm import tqdm
import os
from typing import List, Optional
from span_marker import SpanMarkerModel
from jet.file.utils import load_file, save_file

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set multiprocessing start method to 'spawn' for M1 compatibility
multiprocessing.set_start_method('spawn', force=True)

# Optimize for M1: Disable MPS, limit threading, and disable tokenizer parallelism
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure NLTK data is available
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('maxent_ne_chunker_tab')
# nltk.download('words')


class Result:
    span: str
    label: str
    score: float
    char_start: int
    char_end: int


class POSTaggerProperNouns:
    def __init__(self, output_file: Optional[str] = None):
        self.output_file = output_file
        self.model = None
        logging.info("Initialized POSTaggerProperNouns")

    def load_model(self) -> SpanMarkerModel:
        if self.model:
            logging.debug("Returning cached model")
            return self.model

        model_name = "tomaarsen/span-marker-bert-base-fewnerd-fine-super"
        try:
            logging.info(f"Loading model: {model_name}")
            self.model = SpanMarkerModel.from_pretrained(
                model_name, device='cpu')  # Use CPU to avoid MPS issues
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
        return self.model

    def predict(self, text) -> List[Result]:
        try:
            # Preprocess text to remove problematic characters
            # Remove special characters like '#'
            text = re.sub(r'[^\w\s.,!?]', '', text)
            model = self.load_model()
            logging.debug(f"Predicting on text: {text[:50]}...")
            # Small batch size to reduce memory load
            results = model.predict(
                text, show_progress_bar=False, batch_size=1)
            logging.debug(f"Raw results: {results}")

            # Convert results to dictionary format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'span': result['span'],
                    'label': result['label'],
                    'score': result['score'],
                    'char_start': result['char_start_index'],
                    'char_end': result['char_end_index']
                })
            return formatted_results
        except Exception as e:
            logging.error(f"Error in predict: {e}")
            return []

    def format(self, results, text) -> str:
        try:
            text_with_tags = text
            for result in results:
                span = result['span']
                label = result['label']
                char_start = result['char_start']
                char_end = result['char_end']
                text_with_tags = text_with_tags[:char_start] + \
                    f"{span}/{label}" + text_with_tags[char_end:]
            return text_with_tags
        except Exception as e:
            logging.error(f"Error in format: {e}")
            return text


class SubjectExtractor:
    def extract_subject(self, text):
        """
        Extract the subject from the text.

        Args:
            text (str): The text to analyze.

        Returns:
            list: A list of subjects found in the text, each as a dict with 'text' and 'label'.
        """
        subjects = []
        try:
            sentences = sent_tokenize(text)
            for sentence in sentences:
                words = word_tokenize(sentence)
                pos_tags = pos_tag(words)
                ne_tree = ne_chunk(pos_tags)
                for chunk in ne_tree:
                    if hasattr(chunk, 'label') and chunk.label() in ['PERSON', 'ORGANIZATION', 'GPE', 'LOCATION']:
                        subjects.append({
                            'text': ' '.join(c[0] for c in chunk),
                            'label': chunk.label()
                        })
                    else:
                        if isinstance(chunk, tuple) and chunk[1] in ['NNP', 'PRP']:
                            subjects.append({
                                'text': chunk[0],
                                'label': chunk[1]
                            })
        except Exception as e:
            logging.error(f"Error in extract_subject: {e}")
        return subjects


def get_label(label):
    if label == 'PER':
        return 'Person'
    elif label == 'ORG':
        return 'Organization'
    elif label == 'LOC':
        return 'Location'
    return 'Unknown'


def analyze_subject(text, extractor, model=None):
    try:
        subjects = extractor.extract_subject(text)
        if not subjects:
            logging.warning(f"No subjects found in text: {text[:50]}...")
            return None

        subject = subjects[0]['text']
        logging.debug(f"Processing subject: {subject}")

        if model is None:
            logging.warning("No model provided, returning NLTK-based result")
            return {
                'text': subject,
                'pos': 'PROPN',
                'score': 1.0,
                'label': get_label(subjects[0]['label'])
            }

        pos_ner_results = model.predict(text)
        logging.debug(f"POS/NER results: {pos_ner_results}")

        for item in pos_ner_results:
            matched_pos_words = re.findall(
                rf'\b{re.escape(subject)}\b', item['span'])
            if matched_pos_words:
                return {
                    'text': subject,
                    'pos': 'PROPN',
                    'score': item['score'],
                    'label': get_label(item['label'])
                }
        logging.warning(f"No matching NER results for subject: {subject}")
        return None
    except Exception as e:
        logging.error(f"Error in analyze_subject: {e}")
        return None


def analyze_subjects(texts):
    extractor = SubjectExtractor()
    model = None
    try:
        logging.info("Loading POSTaggerProperNouns model")
        model = POSTaggerProperNouns()
    except Exception as e:
        logging.error(
            f"Error loading POSTaggerProperNouns: {e}, proceeding without model")
        model = None

    results = []
    for text in tqdm(texts, desc="Analyzing subjects"):
        result = analyze_subject(text, extractor, model)
        if result:
            results.append(result)

    return results


if __name__ == '__main__':
    # Define file paths
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )

    # Load texts
    try:
        docs = load_file(docs_file)
        logging.info(f"Loaded JSON data {len(docs)} from: {docs_file}")
        texts = [doc["text"] for doc in docs]
    except Exception as e:
        logging.error(f"Error loading docs: {e}")
        texts = [
            'Manila, officially the City of Manila, is the capital and second-most populous city of the Philippines.'
        ]

    # Analyze subjects
    results = analyze_subjects(texts)

    # Save results
    logging.info(f"Results: {len(results)}")
    os.makedirs(output_dir, exist_ok=True)
    save_file(results, f"{output_dir}/results.json")

    # Clean up multiprocessing resources
    logging.debug("Cleaning up multiprocessing resources")
    for proc in multiprocessing.active_children():
        proc.terminate()
        proc.join()
    logging.debug("Multiprocessing cleanup complete")
