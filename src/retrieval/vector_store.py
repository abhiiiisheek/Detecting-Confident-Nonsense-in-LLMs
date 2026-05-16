import faiss
import numpy as np

dimension = 384

index = faiss.IndexFlatL2(dimension)
documents = []


def add_documents(docs):
    global documents

    embeddings = np.array(docs["embeddings"]).astype("float32")
    index.add(embeddings)

    documents.extend(docs["texts"])


def search(query_embedding, k=3):
    query_embedding = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_embedding, k)

    return [documents[i] for i in indices[0]]
