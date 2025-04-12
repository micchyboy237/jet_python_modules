You are given a user query, some general instructions, all inside xml tags. You have to answer the query based on the instructions

<user_query>
{query}
</user_query>

<general_instructions>
Design a simple and minimal JSON schema that covers the key data points based on the user query. Ensure the schema is intuitive, easy to parse, and includes both required and optional fields based on the typical structure expected for this kind of data. The schema should capture the most important details without overcomplicating the structure.

If the query indicates a collection or multiple results (e.g., "Top", "List", "Multiple"), ensure that the schema uses an array (or list) to represent those results.

Output ONLY the JSON schema wrapped in a `json` code block.
</general_instructions>
