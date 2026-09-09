import logging

from jet.adapters.unstructured.html_schema_generator import (
    generate_html_schema,
    schema_to_llm_context,
)

logging.basicConfig(level=logging.INFO)

schema = generate_html_schema(
    "<html><body><h1>FAQ</h1><p>Answer here.</p></body></html>"
)
llm_context = schema_to_llm_context(schema)
print(llm_context)
# [Title] FAQ
#   [NarrativeText] Answer here.
