from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="gpt2",  
    device=-1       
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
