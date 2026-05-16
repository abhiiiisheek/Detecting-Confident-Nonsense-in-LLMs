def compute_trust_score(analysis, entropy, retrieved_docs, verified_docs=None):
    try:
        # --- Extract values ---
        hallucination = analysis["hallucination_score"]
        avg_similarity = analysis["avg_similarity"]
        overlap = analysis["keyword_overlap"]
        length_penalty = analysis.get("length_penalty", 0)

        entropy_value = entropy["entropy"]

        # --- Base scores ---
        grounding_score = avg_similarity
        overlap_score = overlap
        reliability_score = 1 - hallucination

        # --- PQC (or verification) score ---
        if verified_docs is not None and len(retrieved_docs) > 0:
            pqc_score = len(verified_docs) / len(retrieved_docs)
        else:
            pqc_score = 1.0

        # --- Entropy logic ---
        if overlap_score < 0.3:
            entropy_score = min(1.0, entropy_value / 6.0)
        else:
            entropy_score = max(0.0, 1 - (entropy_value / 6.0))

        # --- Base trust score (compute FIRST) ---
        trust_score = (
            0.2 * grounding_score +
            0.25 * overlap_score +
            0.2 * reliability_score +
            0.15 * pqc_score +
            0.2 * entropy_score
        )

        # --- Penalties (apply AFTER computing trust_score) ---
        # Hard penalty for very low overlap
        if overlap_score < 0.1:
            trust_score *= 0.6

        # Penalize long / rambling outputs
        if length_penalty > 1.5:
            trust_score *= 0.8

        # Clamp to [0, 1]
        trust_score = max(0.0, min(1.0, trust_score))

        return {
            "trust_score": float(trust_score),
            "components": {
                "grounding": grounding_score,
                "overlap": overlap_score,
                "reliability": reliability_score,
                "pqc": pqc_score,
                "entropy": entropy_score,
                "length_penalty": length_penalty
            }
        }

    except Exception as e:
        return {"error": str(e)}
