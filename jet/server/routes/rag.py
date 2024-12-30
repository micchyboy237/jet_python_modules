from pydantic import BaseModel
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from helpers.rag import RAG
from script_utils import display_source_nodes
from jet.transformers import make_serializable
from jet.logger import logger

router = APIRouter()

# Create default RAG instance (will be updated in the endpoint)
rag_global: RAG = None


# Define the schema for input queries
class QueryRequest(BaseModel):
    query: str
    data_dir: str = "/Users/jethroestrada/Desktop/External_Projects/JetScripts/llm/eval/converted-notebooks/retrievers/data/jet-resume"
    rag_dir: str = "/Users/jethroestrada/Desktop/External_Projects/jet_python_modules/jet/server/rag/generated/summaries"
    extensions: list[str] = [".md"]
    system: str = (
        "You are a job applicant providing tailored responses during an interview.\n"
        "Always answer questions using the provided context as if it is your resume, "
        "and avoid referencing the context directly.\n"
        "Some rules to follow:\n"
        "1. Never directly mention the context or say 'According to my resume' or similar phrases.\n"
        "2. Provide responses as if you are the individual described in the context, focusing on professionalism and relevance."
    )


def setup_rag(
    system: str,
    data_dir: str,
    rag_dir: str,
    extensions: list[str],
):
    global rag_global
    rag_global = RAG(system, data_dir, rag_dir, extensions)
    return rag_global


def event_stream(query):
    """Generator function to yield events for streaming."""
    for chunk in rag_global.query(query):
        yield f"data: {chunk}\n\n"


@router.post("/query")
async def query(query_request: QueryRequest):
    global rag_global

    query = query_request.query.strip()

    # Initialize or update RAG instance based on incoming parameters
    rag_global = setup_rag(query_request.system, query_request.data_dir,
                           query_request.rag_dir, query_request.extensions)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
    }
    return StreamingResponse(event_stream(query), headers=headers)


@router.post("/nodes")
async def get_nodes(query_request: QueryRequest):
    global rag_global

    query = query_request.query.strip()

    # Initialize or update RAG instance based on incoming parameters
    rag_global = setup_rag(query_request.system, query_request.data_dir,
                           query_request.rag_dir, query_request.extensions)

    result = rag_global.get_results(query)
    display_source_nodes(query, result["nodes"])
    return make_serializable(result["nodes"])
