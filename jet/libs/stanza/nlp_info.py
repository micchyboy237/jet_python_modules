import os
from typing import Dict, List, Optional
import stanza
from jet.logger import logger

class NLPInfo:
    """A reusable class for processing text with Stanza NLP pipelines."""
    
    def __init__(self, lang: str = "en", model_dir: Optional[str] = None, use_gpu: bool = True):
        """Initialize NLPInfo with language and model directory."""
        self.lang = lang
        self.model_dir = model_dir or os.getenv(
            'STANZA_RESOURCES_DIR',
            os.path.join(os.path.expanduser("~/.cache"), "stanza_resources")
        )
        self.use_gpu = use_gpu
        self.pipelines: Dict[str, stanza.Pipeline] = {}

    def _get_pipeline(self, processors: str) -> stanza.Pipeline:
        """Initialize or retrieve cached Stanza pipeline for given processors."""
        if processors not in self.pipelines:
            logger.info(f"Initializing pipeline for processors: {processors}")
            self.pipelines[processors] = stanza.Pipeline(
                lang=self.lang,
                dir=self.model_dir,
                processors=processors,
                use_gpu=self.use_gpu
            )
        return self.pipelines[processors]

    def tokenize(self, text: str) -> Dict[str, List]:
        """Perform tokenization and sentence segmentation."""
        nlp = self._get_pipeline("tokenize")
        doc = nlp(text)
        sentences = []
        tokens_list = []
        for sent in doc.sentences:
            tokens = [token.text for token in sent.tokens]
            sentences.append(' '.join(tokens))
            tokens_list.append(tokens)
        return {"sentences": sentences, "tokens": tokens_list}

    def mwt(self, text: str) -> Dict[str, List]:
        """Perform multi-word token expansion."""
        nlp = self._get_pipeline("tokenize,mwt")
        doc = nlp(text)
        tokens_per_sentence = []
        words_per_sentence = []
        for sent in doc.sentences:
            tokens = [token.text for token in sent.tokens]
            words = [word.text for word in sent.words]
            tokens_per_sentence.append(tokens)
            words_per_sentence.append(words)
        return {"tokens": tokens_per_sentence, "words": words_per_sentence}

    def pos(self, text: str) -> Dict[str, List]:
        """Perform part-of-speech tagging."""
        nlp = self._get_pipeline("tokenize,mwt,pos")
        doc = nlp(text)
        sentences = []
        pos_tags_list = []
        for sent in doc.sentences:
            sentence_text = ' '.join(word.text for word in sent.words)
            pos_tags = [(word.text, word.upos, word.xpos, word.feats) for word in sent.words]
            sentences.append(sentence_text)
            pos_tags_list.append(pos_tags)
        return {"sentences": sentences, "pos_tags": pos_tags_list}

    def lemma(self, text: str) -> Dict[str, List]:
        """Perform lemmatization."""
        nlp = self._get_pipeline("tokenize,mwt,pos,lemma")
        doc = nlp(text)
        sentences = []
        lemmas_list = []
        for sent in doc.sentences:
            sentence_text = ' '.join(word.text for word in sent.words)
            lemmas = [(word.text, word.lemma) for word in sent.words]
            sentences.append(sentence_text)
            lemmas_list.append(lemmas)
        return {"sentences": sentences, "lemmas": lemmas_list}

    def depparse(self, text: str) -> Dict[str, List]:
        """Perform dependency parsing."""
        nlp = self._get_pipeline("tokenize,mwt,pos,lemma,depparse")
        doc = nlp(text)
        sentences = []
        dependencies_list = []
        for sent in doc.sentences:
            sentence_text = ' '.join(word.text for word in sent.words)
            deps = [(word.text, word.deprel, word.head) for word in sent.words]
            sentences.append(sentence_text)
            dependencies_list.append(deps)
        return {"sentences": sentences, "dependencies": dependencies_list}

    def ner(self, text: str) -> Dict[str, List]:
        """Perform named entity recognition."""
        nlp = self._get_pipeline("tokenize,mwt,ner")
        doc = nlp(text)
        sentences = []
        entities_list = []
        for sent in doc.sentences:
            sentence_text = ' '.join(word.text for word in sent.words)
            entities = [(ent.text, ent.type) for ent in sent.ents]
            sentences.append(sentence_text)
            entities_list.append(entities)
        return {"sentences": sentences, "entities": entities_list}

    def sentiment(self, text: str) -> Dict[str, List]:
        """Perform sentiment analysis."""
        nlp = self._get_pipeline("tokenize,mwt,sentiment")
        doc = nlp(text)
        sentences = []
        sentiment_list = []
        for sent in doc.sentences:
            sentence = ' '.join(word.text for word in sent.words)
            sentiment = sent.sentiment
            sentences.append(sentence)
            sentiment_list.append({
                "value": sentiment,
                "label": ["negative", "neutral", "positive"][sentiment]
            })
        return {"sentences": sentences, "sentiment": sentiment_list}

    def constituency(self, text: str) -> Dict[str, List]:
        """Perform constituency parsing."""
        nlp = self._get_pipeline("tokenize,mwt,pos,constituency")
        doc = nlp(text)
        sentences = []
        parse_trees = []
        for sent in doc.sentences:
            sentence_text = ' '.join(word.text for word in sent.words)
            sentences.append(sentence_text)
            parse_trees.append(str(sent.constituency))
        return {"sentences": sentences, "constituency_trees": parse_trees}
