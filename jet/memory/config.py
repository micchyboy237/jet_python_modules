from langchain_core.prompts import PromptTemplate

CYPHER_QUERY_TEMPLATE = """
Your task is to directly translate natural language inquiry into precise and executable Cypher queries for Memgraph database. 
You will utilize a provided database schema to understand the structure, nodes and relationships within the Memgraph database.
Instructions: 
- Use provided node and relationship labels and property names from the
schema which describes the database's structure. Upon receiving a user
question, synthesize the schema to craft precise Cypher queries that
directly corresponds to the user's intent. 
- Generate multiple valid executable Cypher queries on top of Memgraph database. 
Any explanation, context, or additional information that is not a part 
of the Cypher query syntax should be omitted entirely. 
- Use Memgraph MAGE procedures instead of Neo4j APOC procedures. 
- Do not include any explanations or apologies in your responses. 
- Do not include any text except the generated Cypher statement.
- For queries that ask for information or functionalities outside the direct
generation of Cypher queries, use the Cypher query format to communicate
limitations or capabilities. For example: RETURN "I am designed to generate
Cypher queries based on the provided schema only."

Use pattern matching (CONTAINS) instead of equals. For example:
Instead of:
MATCH (p:Person {{name: "Jethro"}})
RETURN p;
Use this:
MATCH (p:Person)
WHERE toLower(p.name) CONTAINS "jethro"
RETURN p;

Use valid markdown syntax for the response.
<start_schema>
{schema}
<end_schema>

Sample response format:
<start_response>
# Query 1
```cypher
MATCH (p:Person)-[:STUDIED_AT]->(e:Education)
WHERE toLower(p.name) CONTAINS "Jethro"
RETURN e.school, e.degree, e.start_year, e.end_year;
```
---
# Query 2
```cypher
MATCH (c:Company)-[:OWNS]->(proj:Project)-[:PUBLISHED_AT]->(port:Portfolio_Link)
WHERE toLower(c.name) CONTAINS "adec innovations" AND toLower(port.url) CONTAINS "adec"
RETURN proj.name, port.url;
```
<end_response>

With all the above information and instructions, generate {num_of_queries} Cypher queries that will provide sensible data based on the user prompt.

Prompt:
{prompt}
Response:
<start_response>
""".strip()

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"], template=CYPHER_QUERY_TEMPLATE
)


CONTEXT_QA_TEMPLATE = """Your task is to form nice and human understandable answers. The context contains the cypher query result that you must use to construct an answer. The provided context is authoritative, you must never doubt it or try to use your internal knowledge to correct it. Make the answer sound as a response to the question. Do not mention that you based the result on the given context. Here is an example:

Question: Which managers own Neo4j stocks?
Context: [manager:CTL LLC, manager:JANE STREET GROUP LLC]
Helpful Answer: CTL LLC, JANE STREET GROUP LLC owns Neo4j stocks.

Follow this example when generating answers. If the provided context is empty, say that you don't know the answer.
Write the answer in the tone of {tone_name}.


Question: {question}
Context:
{context}
Helpful Answer:"""

CONTEXT_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=CONTEXT_QA_TEMPLATE
)


CONTEXT_SAMPLES_TEMPLATE = """
Sample Queries:
{sample_queries_str}
""".strip()

CONTEXT_DB_TEMPLATE = """
<start_db_results>
{db_results_str}
<end_db_results>
""".strip()

CONTEXT_SCHEMA_TEMPLATE = """
<start_schema>
{schema_str}
<end_schema>
""".strip()
