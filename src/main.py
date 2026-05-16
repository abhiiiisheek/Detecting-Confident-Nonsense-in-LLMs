from fastapi import FastAPI
from src.api.routes import router
from src.retrieval.embedder import get_embedding
from src.retrieval.vector_store import add_documents


from src.retrieval.embedder import get_embedding
from src.retrieval.vector_store import add_documents

docs = [
    "Artificial Intelligence is the simulation of human intelligence by machines.",
    "Machine learning is a subset of AI focused on learning from data.",
    "AI is used in healthcare, robotics, and natural language processing."
]

embeddings = [get_embedding(doc) for doc in docs]

add_documents({
    "texts": docs,
    "embeddings": embeddings
})

embeddings = [get_embedding(doc) for doc in docs]

add_documents({
    "texts": docs,
    "embeddings": embeddings
})

app = FastAPI(title="Secure LLM Alignment System")


app.include_router(router)
