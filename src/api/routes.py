from fastapi import APIRouter
from src.llm.generator import generate_response
from src.scoring.trust_score import compute_trust_score
from src.interpretability.entropy import compute_entropy

router = APIRouter()


@router.post("/generate")
def generate(prompt: str):
    result = generate_response(prompt)
    return result
    
    
from src.retrieval.retriever import retrieve

@router.post("/test-retrieval")
def test(prompt: str):
    docs = retrieve(prompt)
    return {"retrieved_docs": docs}

from src.retrieval.retriever import retrieve
from src.scoring.hallucination import compute_hallucination_score

@router.post("/analyze")
def analyze(prompt: str):
    response = generate_response(prompt)

    retrieved_docs = retrieve(prompt + " " + response["response"])

    analysis = compute_hallucination_score(
        response["response"],
        retrieved_docs
    )

    entropy = compute_entropy(response["response"])

    verified_docs = retrieved_docs  # PQC placeholder

    trust = compute_trust_score(
        analysis,
        entropy,
        retrieved_docs,
        verified_docs
    )

    return {
        "llm_output": response["response"],
        "retrieved_docs": retrieved_docs,
        "analysis": analysis,
        "entropy": entropy,
        "trust": trust
    }
    
