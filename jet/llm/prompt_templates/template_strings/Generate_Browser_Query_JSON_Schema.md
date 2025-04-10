You are given a browser query. Based on the query, generate a simple and concise JSON schema that represents the minimum necessary data structure. The schema should include the required fields, as well as any optional fields that are commonly expected based on the user's query. Fields should be kept to the essential ones for the query, using validation where applicable (e.g., `required`, `type`, etc.). Optional fields should be included but not marked as `required`.

If the query suggests multiple items (like a list of results), ensure that the schema reflects this by using an array (`type: array`) for those properties.

<browser_query>
{browser_query}
</browser_query>

<user_query>
Design a simple and minimal JSON schema that covers the key data points based on the browser query. Ensure the schema is intuitive, easy to parse, and includes both required and optional fields based on the typical structure expected for this kind of data. The schema should capture the most important details without overcomplicating the structure.

If the query indicates a collection or multiple results (e.g., "Top", "List", "Multiple"), ensure that the schema uses an array (or list) to represent those results.

The output should follow this format:

<json_query_and_schema_sample>
<browser_query>
Top otome villainess anime 2025
</browser_query>

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
</json_query_and_schema_sample>

Output ONLY the JSON schema wrapped in a `json` code block.
</user_query>
