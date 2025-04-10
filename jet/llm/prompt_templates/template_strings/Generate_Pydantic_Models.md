You are given a user query, some textual context, all inside xml tags. You have to answer the query based on the context.

<context>
{context}
</context>

<user_query>
Design a set of robust Pydantic models to represent application data that can be extracted from the provided context. The models should ensure data validation, consistency, and ease of extensibility. The models must reflect a comprehensive structure, capturing all relevant aspects from the context, including hierarchical nesting where applicable.

Use appropriate Pydantic field types and leverage features such as:

- `Field()` with descriptions and defaults or constraints
- `Optional[]` for nullable fields
- `constr`, `conint`, `condecimal`, etc., for type constraints
- Nested `BaseModel` classes to mirror structured or list-based data

Include a sample schema structure with descriptions so an LLM or human can understand expected output easily.

### Output Example:

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class Answer(BaseModel):
    title: str = Field(
        ...,
        description="The exact title of the anime, as it appears in the document."
    )
    document_number: int = Field(
        ...,
        description="The number of the document that includes this anime (e.g., 'Document number: 3')."
    )
    release_year: Optional[int] = Field(
        None,
        description="The most recent known release year of the anime, if specified in the document."
    )

class QueryResponse(BaseModel):
    results: List[Answer] = Field(
        default_factory=list,
        description="List of relevant anime titles extracted from the documents, matching the user's query.\nEach entry includes the title, source document number, and release year (if known)."
    )
```

The final models should be structured similarly to the above example, tailored to the given context. Output ONLY the Python code wrapped in a `python` code block.
</user_query>
