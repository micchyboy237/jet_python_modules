import os
import shutil

from jet.features.web_retrieval_project.src.scraper import scrape_recursive_url
from jet.features.web_retrieval_project.src.indexer import index_scraped_docs
from jet.features.web_retrieval_project.src.retriever import rag_query, RAGInput
from jet.file.utils import save_file
from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

if __name__ == "__main__":
    query = "What is Python's garbage collector?"
    url = "https://docs.python.org/3.12/"
    docs = scrape_recursive_url(url, max_depth=1)
    vectorstore = index_scraped_docs(docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    answer = rag_query({"query": query, "retriever": retriever})

    save_file({
        "query": query,
        "url": url,
    }, f"{OUTPUT_DIR}/input.json")
    save_file(answer, f"{OUTPUT_DIR}/answer.md")
