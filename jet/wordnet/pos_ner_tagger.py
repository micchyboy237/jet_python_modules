from span_marker import SpanMarkerModel
from jet.logger import time_it
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
        self.model = None

    @time_it
    def load_model(self) -> SpanMarkerModel:
        if self.model:
            return self.model

        model_name = "tomaarsen/span-marker-bert-base-fewnerd-fine-super"
        self.model = SpanMarkerModel.from_pretrained(model_name)
        return self.model

    @time_it
    def predict(self, text) -> List[Result]:
        model = self.load_model()

        results = model.predict(text, show_progress_bar=True)

        # Replace char_start and char_end with actual values
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
            text_with_tags = text_with_tags.replace(
                text_with_tags[char_start:char_end], f"{span}/{label}"
            )
        return text_with_tags
