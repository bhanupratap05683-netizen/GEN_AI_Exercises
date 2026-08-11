import tiktoken

def explore_tokenization():
    # GPT-4 encoder
    enc = tiktoken.encoding_for_model("gpt-4o")
    
    test_strings = [
        "Hello world",
        "unhappiness",
        "The quick brown fox",
        "SELECT * FROM users WHERE id = 1",
        "🚀 emoji test",
        "color colour",  # same word different spelling
    ]
    
    for text in test_strings:
        tokens = enc.encode(text)
        decoded = [enc.decode([t]) for t in tokens]
        print(f"Text: '{text}'")
        print(f"Token IDs: {tokens}")
        print(f"Token strings: {decoded}")
        print(f"Token count: {len(tokens)}")
        print("---")

explore_tokenization()