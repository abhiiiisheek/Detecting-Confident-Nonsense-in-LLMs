# src/interpretability/entropy.py

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model + tokenizer once
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def compute_entropy(text: str):
    try:
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Shift logits for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # Convert logits → probabilities
        probs = F.softmax(shift_logits, dim=-1)

        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        avg_entropy = torch.mean(entropy).item()

        return {
            "entropy": avg_entropy
        }

    except Exception as e:
        return {"error": str(e)}
