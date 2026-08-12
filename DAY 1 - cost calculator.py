def calculate_api_cost(text: str, model: str = "gpt-4o") -> dict:
    import tiktoken
    
    PRICES = {
        "gpt-4o": {"input": 0.000005, "output": 0.000015},
        "gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},
        "gpt-3.5-turbo": {"input": 0.0000005, "output": 0.0000015},
    }
    
    enc = tiktoken.encoding_for_model("gpt-4o")
    token_count = len(enc.encode(text))
    
    cost = token_count * PRICES[model]["input"]
    
    return {
        "text_length": len(text),
        "token_count": token_count,
        "model": model,
        "estimated_input_cost": f"${cost:.6f}",
        "tokens_per_char": token_count / len(text)
    }

# Test with different text types
texts = [
    "Simple short sentence.",
    "A" * 1000,  # repetitive text
    "SELECT * FROM transactions WHERE amount > 1000 AND date > '2024-01-01'",
]

for text in texts:
    print(calculate_api_cost(text))