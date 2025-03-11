from span_marker import SpanMarkerModel
from instruction_generator.utils.time_utils import time_it
from instruction_generator.utils.language import detect_lang
from typing import List, Optional


class Result:
    span: str
    label: str
    score: float
    char_start: int
    char_end: int


class POSTaggerProperNouns:
    def __init__(self, output_file: Optional[str] = None):
        self.output_file = output_file
        self.model = {}

    @time_it
    def load_model(self, lang) -> SpanMarkerModel:
        if self.model.get(lang):
            return self.model[lang]

        model_name = "tomaarsen/span-marker-mbert-base-tlunified"

        if lang == "tl":
            model_name = "tomaarsen/span-marker-roberta-tagalog-base-tlunified"

        self.model[lang] = SpanMarkerModel.from_pretrained(model_name)

        return self.model[lang]

    @time_it
    def predict(self, text, lang: Optional[str] = None) -> List[Result]:
        if not lang:
            lang_result = detect_lang(text)
            lang = lang_result.get('lang', 'en')

        model = self.load_model(lang)

        results = model.predict(
            text,
            show_progress_bar=True)

        # Replace char_start and char_end with the actual values
        for result in results:
            result['char_start'] = result['char_start_index']
            result['char_end'] = result['char_end_index']
            del result['char_start_index']
            del result['char_end_index']
        return results

    @time_it
    def format(self, results, text) -> str:
        text_with_tags = text
        for result in results:
            span = result['span']
            label = result['label']
            char_start = result['char_start']
            char_end = result['char_end']

            # Replace text_with_tags with span/PROPN-label
            # Use char_start and char_end to replace the span
            text_with_tags = text_with_tags.replace(
                text_with_tags[char_start:char_end], f"{span}/{label}")
        return text_with_tags
