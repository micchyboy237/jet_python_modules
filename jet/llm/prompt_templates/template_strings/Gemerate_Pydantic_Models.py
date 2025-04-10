You are given a user query, some textual context, all inside xml tags. You have to answer the query based on the context.

<context>
{context}
</context>

<user_query>
Design a set of robust Pydantic models to represent application data that can be extracted from the provided context. The models should ensure data validation, consistency, and ease of extensibility. The models must reflect a comprehensive structure, capturing all relevant aspects from the context, including hierarchical nesting where applicable.

The models should use the appropriate Pydantic field types for each context field, and should leverage Pydantic's features such as `Field` for validation, `Optional` for optional fields, and `constr`, `conint`, `condecimal`, etc., for additional constraints on data types. Ensure that fields which are required are defined without `Optional`.

Fields that are nested in other fields or lists should be modeled as Pydantic `BaseModel` classes, with attributes and types reflecting the nested structure. Provide clear model definitions that enforce validation rules for all fields, both required and optional, while maintaining backward compatibility for future extensibility.

Ensure the models are ready for integration into any Python-based application using Pydantic.

Output ONLY the Python code for the Pydantic models wrapped in a `python` code block.
</user_query>
