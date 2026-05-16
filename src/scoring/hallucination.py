from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')


def keyword_overlap(text, docs):
    text_words = set(text.lower().split())
    doc_words = set(" ".join(docs).lower().split())

    if len(text_words) == 0:
        return 0

    overlap = len(text_words & doc_words) / len(text_words)
    return overlap

def repetition_penalty(text):
    words = text.lower().split()
    unique_words = set(words)
    
    return 1 - (len(unique_words) / len(words))

def compute_hallucination_score(llm_output, retrieved_docs):
    try:
        # --- Step 1: Embeddings ---
        output_embedding = model.encode([llm_output])
        docs_embedding = model.encode(retrieved_docs)
        
        rep_penalty = repetition_penalty(llm_output)
        

        # --- Step 2: Similarity ---
        similarities = cosine_similarity(output_embedding, docs_embedding)

        avg_similarity = float(similarities.mean())
        max_similarity = float(similarities.max())

        # --- Step 3: Keyword overlap ---
        overlap = keyword_overlap(llm_output, retrieved_docs)

        # --- Step 4: Length penalty ---
        length_penalty = len(llm_output.split()) / 100

        # --- Step 5: Final hallucination score ---
        base_score = 1 - avg_similarity
        hallucination_score = base_score * (1 + 0.2 * length_penalty)

        # --- Step 6: Combined (more robust) score ---
        final_score = (  0.6 * (1 - avg_similarity) + 0.2 * (1 - overlap) +  0.2 * rep_penalty)
    

        return {
            "avg_similarity": avg_similarity,
            "max_similarity": max_similarity,
            "keyword_overlap": overlap,
            "length_penalty": length_penalty,
            "hallucination_score": float(hallucination_score),
            "final_score": float(final_score)
        }

    except Exception as e:
        return {"error": str(e)}
