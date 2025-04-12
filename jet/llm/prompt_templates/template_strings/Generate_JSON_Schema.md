You are given a user query, some general instructions, all inside xml tags. You have to answer the query based on the instructions

<user_query>
{query}
</user_query>

<general_instructions>
Design a simple and minimal JSON schema that covers the key data points based on the user query. Ensure the schema is intuitive, easy to parse, and includes both required and optional fields based on the typical structure expected for this kind of data. The schema should capture the most important details without overcomplicating the structure.

If the query indicates a collection or multiple results (e.g., "Top", "List", "Multiple"), ensure that the schema uses an array (or list) to represent those results.

The output should follow this format:

<json_query_schema_response_sample>
<user_query>
Top otome villainess anime 2025
</user_query>

<json_schema>
{{
    "type": "object",
    "properties": {{
        "anime_list": {{
        "type": "array",
        "items": {{
            "type": "object",
            "properties": {{
                "title": {{
                    "type": "string",
                    "description": "The title of the anime."
                }},
                "release_year": {{
                    "type": "integer",
                    "description": "The release year of the anime."
                }},
                "genre": {{
                    "type": "array",
                    "items": {{
                        "type": "string"
                    }},
                    "description": "Genres associated with the anime."
                }},
                "rating": {{
                    "type": "number",
                    "description": "Average user rating of the anime.",
                }}
            }},
            "required": ["title", "release_year"],
            "optional": ["genre", "rating"]
        }},
        "description": "List of top otome villainess anime for 2025."
        }}
    }},
    "required": ["anime_list"]
}}
</json_schema>

<response>
```json
{
    "anime_list": [
    {
        "title": "Villainess: A Reversal of Fortune",
        "release_year": 2025,
        "genre": ["Fantasy", "Romance", "Drama"],
        "rating": 8.7
    },
    {
        "title": "The Villainess Strikes Back",
        "release_year": 2025,
        "genre": ["Fantasy", "Adventure", "Action"],
        "rating": 8.2
    },
    {
        "title": "My Sweet Villainess",
        "release_year": 2025,
        "genre": ["Romance", "Comedy", "Drama"],
        "rating": 7.9
    }
    ]
}
```
</response>
</json_query_schema_response_sample>

Output ONLY the JSON schema wrapped in a `json` code block.
</general_instructions>
