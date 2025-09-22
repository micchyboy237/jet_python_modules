import pandas as pd
import requests
import os
import faiss  # Added for FAISS index creation

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from langchain_openai import OpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate


def import_data(pages: int, start_year: int, end_year: int, search_terms: str) -> pd.DataFrame:
    """
    This function is used to use the OpenAlex API, conduct a search on works, and return a dataframe with associated works.
    
    Inputs: 
        - pages: int, number of pages to loop through
        - search_terms: str, keywords to search for (must be formatted according to OpenAlex standards)
        - start_year and end_year: int, years to set as a range for filtering works
    """

    # create an empty dataframe
    search_results = pd.DataFrame()

    for page in range(1, pages):

        # use parameters to conduct request and format to a dataframe
        response = requests.get(f'https://api.openalex.org/works?page={page}&per-page=200&filter=publication_year:{start_year}-{end_year},type:article&search={search_terms}')
        data = pd.DataFrame(response.json()['results'])

        # append to empty dataframe
        search_results = pd.concat([search_results, data])

    # subset to relevant features
    search_results = search_results[["id", "title", "display_name", "publication_year", "publication_date",
                                        "type", "countries_distinct_count","institutions_distinct_count",
                                        "has_fulltext", "cited_by_count", "keywords", "referenced_works_count", "abstract_inverted_index"]]

    return search_results


def undo_inverted_index(inverted_index: dict) -> str:
    """
    The purpose of the function is to 'undo' an inverted index. It inputs an inverted index and
    returns the original string.
    """

    # create empty lists to store uninverted index
    word_index = []
    words_unindexed = []

    # loop through index and return key-value pairs
    for k, v in inverted_index.items():
        for index in v: word_index.append([k, index])

    # sort by the index
    word_index = sorted(word_index, key=lambda x: x[1])

    # join only the values and flatten
    for pair in word_index:
        words_unindexed.append(pair[0])
    words_unindexed = ' '.join(words_unindexed)

    return words_unindexed


def create_vector_store(ai_search: pd.DataFrame) -> FAISS:
    """
    Creates a FAISS vector store from the provided DataFrame of AI research abstracts.
    
    Args:
        ai_search: DataFrame with 'original_abstract', 'title', and 'publication_year' columns.
    
    Returns:
        FAISS vector store instance.
    """
    # load embedding model
    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")

    # save index with faiss
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    # format abstracts as documents
    documents = [
        Document(
            page_content=ai_search['original_abstract'][i],
            metadata={"title": ai_search['title'][i], "year": ai_search['publication_year'][i]}
        )
        for i in range(len(ai_search))
    ]

    # create list of ids as strings (fixed: use range instead of undefined my_list)
    n = len(ai_search)
    ids = [str(x) for x in range(1, n + 1)]

    # create vector store (fixed: initialize with index and embeddings)
    vector_store = FAISS(
        embedding_function=embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore({id_: doc for id_, doc in zip(ids, documents)}),
        index_to_docstore_id={i: id_ for i, id_ in enumerate(ids)}
    )

    # add documents to vector store
    vector_store.add_documents(documents=documents, ids=ids)

    # save the vector store
    os.makedirs("Data", exist_ok=True)
    vector_store.save_local("Data/faiss_index")

    return vector_store


def setup_rag_pipeline(vector_store: FAISS, openai_api_key: str) -> RetrievalQA:
    """
    Sets up the RAG pipeline using the vector store and OpenAI LLM.
    
    Args:
        vector_store: FAISS vector store.
        openai_api_key: OpenAI API key.
    
    Returns:
        RetrievalQA chain.
    """
    # set API key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # load llm
    llm = OpenAI(openai_api_key=openai_api_key)

    # test that vector database is working (example query)
    print(vector_store.similarity_search("computer vision", k=3))

    # test llm response (example)
    print(llm.invoke("What are the most recent advancements in computer vision?"))

    # test that vector database is working
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # fixed: use vector_store instead of db

    # create a prompt template
    template = """<|user|>
Relevant information:
{context}

Provide a concise answer to the following question using relevant information provided above:
{question}
If the information above does not answer the question, say that you do not know. Keep answers to 3 sentences or shorter.<|end|>
<|assistant|>"""

    # define prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # create RAG pipeline
    rag = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=True
    )

    return rag


if __name__ == "__main__":
    # Example usage (requires OPENAI_API_KEY env var or passed value)
    # search for AI-related research
    ai_search = import_data(30, 2018, 2025, "'artificial intelligence' OR 'deep learn' OR 'neural net' OR 'natural language processing' OR 'machine learn' OR 'large language models' OR 'small language models'")

    # create 'original_abstract' feature
    ai_search['original_abstract'] = list(map(undo_inverted_index, ai_search['abstract_inverted_index']))

    # Create vector store
    vector_store = create_vector_store(ai_search)

    # Setup RAG (replace with your API key)
    rag = setup_rag_pipeline(vector_store, os.getenv("OPENAI_API_KEY", "your-api-key-here"))

    # Test RAG response (example)
    input_query = 'What are the most recent advancements in computer vision?'
    print(rag.invoke(input_query))
