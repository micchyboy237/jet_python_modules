from typing import List, Optional
from jet.utils.url_utils import clean_url, parse_url
from jet.vectors.document_types import HeaderDocument
from .stratified_sampler import StratifiedSampler, ProcessedDataString


def sample_diverse_headers(
    docs: List[HeaderDocument],
    num_samples: Optional[int] = None,
    n: int = 2,
    top_n: int = 1,
    category_values: Optional[List[List[str]]] = None
) -> List[str]:
    pass
