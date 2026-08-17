# HANDS-ON: Embeddings Explorer

import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

# Part 1: Generate and compare embeddings
async def explore_embeddings():
    client = openai.AsyncOpenAI()
    
    async def embed_openai(texts: list[str]) -> list[list[float]]:
        response = await client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [item.embedding for item in response.data]
    
    def cosine_similarity(a: list, b: list) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    # Test semantic similarity
    texts = [
        "The cat sat on the mat",
        "A feline rested on a rug",   # semantically similar
        "The stock market crashed",    # semantically different
        "Python is a programming language",
        "Snake is a reptile",          # ambiguous - python/snake
    ]
    
    embeddings = await embed_openai(texts)
    
    print("Similarity Matrix:")
    print(f"{'':40}", end="")
    for i in range(len(texts)):
        print(f"T{i+1}   ", end="")
    print()
    
    for i, (text, emb_i) in enumerate(zip(texts, embeddings)):
        print(f"T{i+1}: {text[:38]:38}", end="")
        for j, emb_j in enumerate(embeddings):
            sim = cosine_similarity(emb_i, emb_j)
            print(f"{sim:.2f} ", end="")
        print()

# Part 2: Chunking strategies
def chunk_fixed_size(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def chunk_by_sentences(text: str, sentences_per_chunk: int = 3) -> list[str]:
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
    return chunks

def chunk_recursive(
    text: str, 
    chunk_size: int = 300,
    overlap: int = 50,
    separators: list = ["\n\n", "\n", ". ", " ", ""]
) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""
            
            for part in parts:
                if len(current) + len(part) + len(sep) <= chunk_size:
                    current += part + sep
                else:
                    if current:
                        chunks.append(current.strip())
                        # Keep overlap
                        overlap_text = current[-overlap:] if len(current) > overlap else current
                        current = overlap_text + part + sep
                    else:
                        current = part + sep
            
            if current:
                chunks.append(current.strip())
            
            return [c for c in chunks if c]
    
    return [text]

# Test chunking
sample_text = """
Artificial intelligence has transformed how we interact with technology.
Modern language models can understand context and generate coherent text.

The field has grown rapidly since the introduction of transformer architecture.
Companies now invest billions in AI research and development.

Applications range from chatbots to medical diagnosis tools.
The ethical implications of AI continue to be debated globally.
""".strip()

print("=== Fixed Size Chunks ===")
for i, chunk in enumerate(chunk_fixed_size(sample_text, 100, 20)):
    print(f"Chunk {i+1} ({len(chunk)} chars): {chunk[:50]}...")

print("\n=== Sentence Chunks ===")
for i, chunk in enumerate(chunk_by_sentences(sample_text, 2)):
    print(f"Chunk {i+1}: {chunk[:80]}...")