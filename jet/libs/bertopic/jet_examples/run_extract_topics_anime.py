from typing import List, Dict
from jet.adapters.bertopic import BERTopic
from jet.libs.bertopic.examples.mock import load_sample_data
import spacy
import pandas as pd

from jet.file.utils import save_file
from jet.logger import logger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

class AnimeAnalyzer:
    def __init__(self, model_name: str = "embeddinggemma"):
        """Initialize the analyzer with a BERTopic model and spaCy for NER."""
        self.nlp = spacy.load("en_core_web_sm")
        self.topic_model = BERTopic(embedding_model=model_name, min_topic_size=2)

    def extract_titles(self, texts: List[str]) -> List[str]:
        """Extract potential anime titles using spaCy NER."""
        titles = []
        for text in texts:
            doc = self.nlp(text)
            # Extract entities likely to be titles (e.g., proper nouns)
            for ent in doc.ents:
                if ent.label_ in ["WORK_OF_ART", "PRODUCT"]:
                    titles.append(ent.text)
        return list(set(titles))  # Remove duplicates

    def analyze_genres(self, texts: List[str]) -> List[Dict[str, str]]:
        """Analyze texts to extract topics (genres) and categorize as anime."""
        # Fit BERTopic model to texts
        topics, _ = self.topic_model.fit_transform(texts)
        topic_info = self.topic_model.get_topic_info()
        save_file(topic_info.to_dict(orient="records"), f"{OUTPUT_DIR}/topic_info.json")

        # Extract titles
        titles = self.extract_titles(texts)
        save_file(titles, f"{OUTPUT_DIR}/titles.json")

        results = []
        for idx, (text, topic) in enumerate(zip(texts, topics)):
            # Get topic keywords to infer genre
            topic_words = self.topic_model.get_topic(topic)
            genre = ", ".join([word for word, _ in topic_words[:3]])  # Top 3 words as genre
            # Find matching title in text (simplified matching)
            matching_title = next((title for title in titles if title in text), "Unknown")
            results.append({
                "title": matching_title,
                "genre": genre,
                "category": "anime"
            })

        return results

    def save_results(self, results: List[Dict[str, str]], output_path: str) -> None:
        """Save results to a CSV file."""
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.success(f"Saved CSV file to {output_path}")

# Example usage
if __name__ == "__main__":
    # Sample web-scraped texts
    texts = load_sample_data()
    analyzer = AnimeAnalyzer()
    results = analyzer.analyze_genres(texts)
    analyzer.save_results(results,  f"{OUTPUT_DIR}/anime_results.csv")
    save_file(results, f"{OUTPUT_DIR}/genres.json")