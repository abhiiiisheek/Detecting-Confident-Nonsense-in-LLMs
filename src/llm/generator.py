from transformers import pipeline

# Load model once (important)
generator = pipeline(
    "text-generation",
    model="gpt2",   # start simple
    device=-1       # CPU (use 0 if GPU available)
)

def generate_response(prompt: str) -> dict:
    try:
        result = generator(
            prompt,
            max_length=100,
            num_return_sequences=1
        )

        output_text = result[0]["generated_text"]

        return {
            "prompt": prompt,
            "response": output_text,
            "model": "gpt2-local"
        }

    except Exception as e:
        return {
            "error": str(e)
        }
