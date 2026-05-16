# src/pqc/verifier.py

def verify_document(doc: str):
    trusted_keywords = [
        "Artificial Intelligence",
        "Machine learning",
        "healthcare",
        "robotics"
    ]

    for keyword in trusted_keywords:
        if keyword.lower() in doc.lower():
            return True

    return False
