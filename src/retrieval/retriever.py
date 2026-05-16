from src.retrieval.embedder import get_embedding
from src.retrieval.vector_store import search
from src.pqc.verifier import verify_document


def retrieve(query: str):
    query_embedding = get_embedding(query)
    results = search(query_embedding)

    verified_docs = []

    for doc in results:
        if verify_document(doc):
            verified_docs.append(doc)

    return verified_docs
