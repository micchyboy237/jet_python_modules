You are given a user query, some textual context, all inside xml tags. You have to answer the query based on the context

<context>
{context}
</context>

<user_query>
Design a robust JSON schema to represent application database data that can be extracted from context. The schema must be adaptable for diverse scenarios, ensuring comprehensive coverage of all critical aspects, including required and optional fields. It should include fields to capture details across the context and support hierarchical nesting.

The schema should enforce data validation rules and clearly specify which fields are mandatory (required) and which are optional. This should be done by utilizing the `required` keyword for mandatory fields and omitting it for optional ones. Ensure that the schema is concise, validated, and consistent with modern JSON schema standards.

The design should facilitate easy parsing and future extensibility, ensuring that new fields can be added without breaking backward compatibility.

Output ONLY the JSON schema wrapped in a `json` code block.
</user_query>
