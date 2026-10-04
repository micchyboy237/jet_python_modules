(.venv) jethroestrada@Jethros-Mac-mini ~ % python -c "
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

print('=== OpenInferenceSpanKindValues ===')
for e in OpenInferenceSpanKindValues:
print(f'{e.name} = {e.value}')

print()
print('=== SpanAttributes ===')
for name, value in vars(SpanAttributes).items():
if not name.startswith('\_') and isinstance(value, str):
print(f'{name} = {value}')
"
=== OpenInferenceSpanKindValues ===
TOOL = TOOL
CHAIN = CHAIN
LLM = LLM
RETRIEVER = RETRIEVER
EMBEDDING = EMBEDDING
AGENT = AGENT
RERANKER = RERANKER
UNKNOWN = UNKNOWN
GUARDRAIL = GUARDRAIL
EVALUATOR = EVALUATOR
PROMPT = PROMPT

=== SpanAttributes ===
ANNOTATIONS = annotations
EVALUATIONS = evaluations
TRACE_ANNOTATIONS = trace.annotations
TRACE_EVALUATIONS = trace.evaluations
SESSION_ANNOTATIONS = session.annotations
SESSION_EVALUATIONS = session.evaluations
OUTPUT_VALUE = output.value
OUTPUT_MIME_TYPE = output.mime_type
INPUT_VALUE = input.value
INPUT_MIME_TYPE = input.mime_type
INPUT_IMAGES = input.images
OUTPUT_IMAGES = output.images
EMBEDDING_EMBEDDINGS = embedding.embeddings
EMBEDDING_INVOCATION_PARAMETERS = embedding.invocation_parameters
EMBEDDING_MODEL_NAME = embedding.model_name
LLM_FUNCTION_CALL = llm.function_call
LLM_INVOCATION_PARAMETERS = llm.invocation_parameters
LLM_INPUT_MESSAGES = llm.input_messages
LLM_OUTPUT_MESSAGES = llm.output_messages
LLM_MODEL_NAME = llm.model_name
LLM_REQUEST_MODEL_NAME = llm.request.model_name
LLM_RESPONSE_MODEL_NAME = llm.response.model_name
LLM_PROVIDER = llm.provider
LLM_SYSTEM = llm.system
LLM_PROMPTS = llm.prompts
LLM_CHOICES = llm.choices
LLM_PROMPT_TEMPLATE = llm.prompt_template.template
LLM_PROMPT_TEMPLATE_VARIABLES = llm.prompt_template.variables
LLM_PROMPT_TEMPLATE_VERSION = llm.prompt_template.version
LLM_TOKEN_COUNT_COMPLETION = llm.token_count.completion
LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO = llm.token_count.completion_details.audio
LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING = llm.token_count.completion_details.reasoning
LLM_TOKEN_COUNT_PROMPT = llm.token_count.prompt
LLM_TOKEN_COUNT_PROMPT_DETAILS = llm.token_count.prompt_details
LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO = llm.token_count.prompt_details.audio
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_INPUT = llm.token_count.prompt_details.cache_input
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = llm.token_count.prompt_details.cache_read
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE = llm.token_count.prompt_details.cache_write
LLM_TOKEN_COUNT_TOTAL = llm.token_count.total
LLM_FINISH_REASON = llm.finish_reason
LLM_COST_COMPLETION = llm.cost.completion
LLM_COST_COMPLETION_DETAILS = llm.cost.completion_details
LLM_COST_COMPLETION_DETAILS_AUDIO = llm.cost.completion_details.audio
LLM_COST_COMPLETION_DETAILS_OUTPUT = llm.cost.completion_details.output
LLM_COST_COMPLETION_DETAILS_REASONING = llm.cost.completion_details.reasoning
LLM_COST_PROMPT = llm.cost.prompt
LLM_COST_PROMPT_DETAILS = llm.cost.prompt_details
LLM_COST_PROMPT_DETAILS_AUDIO = llm.cost.prompt_details.audio
LLM_COST_PROMPT_DETAILS_CACHE_INPUT = llm.cost.prompt_details.cache_input
LLM_COST_PROMPT_DETAILS_CACHE_READ = llm.cost.prompt_details.cache_read
LLM_COST_PROMPT_DETAILS_CACHE_WRITE = llm.cost.prompt_details.cache_write
LLM_COST_PROMPT_DETAILS_INPUT = llm.cost.prompt_details.input
LLM_COST_TOTAL = llm.cost.total
LLM_TOOLS = llm.tools
TOOL_NAME = tool.name
TOOL_DESCRIPTION = tool.description
TOOL_PARAMETERS = tool.parameters
TOOL_ID = tool.id
RETRIEVAL_DOCUMENTS = retrieval.documents
METADATA = metadata
TAG_TAGS = tag.tags
OPENINFERENCE_SPAN_KIND = openinference.span.kind
SESSION_ID = session.id
USER_ID = user.id
AGENT_NAME = agent.name
GRAPH_NODE_ID = graph.node.id
GRAPH_NODE_NAME = graph.node.name
GRAPH_NODE_PARENT_ID = graph.node.parent_id
PROMPT_VENDOR = prompt.vendor
PROMPT_ID = prompt.id
PROMPT_URL = prompt.url
(.venv) jethroestrada@Jethros-Mac-mini ~ %
