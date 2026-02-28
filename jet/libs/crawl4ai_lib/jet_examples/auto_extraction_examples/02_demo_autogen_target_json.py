import json
from crawl4ai import JsonCssExtractionStrategy

# When you know exactly what JSON structure you want
target_json_example = """
{
    "name": "Product Name",
    "price": "$99.99",
    "rating": 4.5,
    "features": ["feature1", "feature2"],
    "description": "Product description"
}
"""

# sample_html and llm_config are assumed to be obtained beforehand
# schema = JsonCssExtractionStrategy.generate_schema(
#     html=sample_html,
#     target_json_example=target_json_example,
#     llm_config=llm_config
# )
