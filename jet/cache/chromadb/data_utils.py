import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Any
from jet.logger import logger


def connect_chromadb(persistence_path: str, collection_name: str) -> chromadb.Collection:
    """Connect to ChromaDB collection."""
    try:
        client = chromadb.PersistentClient(
            path=persistence_path,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(collection_name)
        logger.info(
            f"Successfully connected to ChromaDB collection: {collection_name}")
        return collection
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {str(e)}")
        raise


def get_chromadb_data(collection: chromadb.Collection, limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve data from ChromaDB collection."""
    try:
        results = collection.get(limit=limit)
        logger.success(f"Retrieved {len(results['documents'])} documents")
        return results
    except Exception as e:
        logger.error(f"Error retrieving data: {str(e)}")
        return {"documents": [], "metadatas": []}


def check_chromadb_data(persistence_path: str = str(Path.home() / ".chromadb_autogen"),
                        collection_name: str = "autogen_docs") -> None:
    """Verify data stored in ChromaDBVectorMemory."""
    collection = connect_chromadb(persistence_path, collection_name)
    data = get_chromadb_data(collection)

    expected_contents = [
        "The weather should be in metric units",
        "Meal recipe must be vegan"
    ]
    expected_metadata = [
        {"category": "preferences", "type": "units"},
        {"category": "preferences", "type": "dietary"}
    ]

    documents = data["documents"]
    metadatas = data["metadatas"]

    for content, metadata in zip(expected_contents, expected_metadata):
        if content in documents:
            logger.success(f"Found expected content: {content}")
            content_index = documents.index(content)
            found_metadata = metadatas[content_index]
            if found_metadata == metadata:
                logger.success(
                    f"Metadata matches expected: {json.dumps(metadata)}")
            else:
                logger.warning(
                    f"Metadata mismatch for {content}: got {json.dumps(found_metadata)}, expected {json.dumps(metadata)}")
        else:
            logger.warning(f"Expected content not found: {content}")
