# HANDS-ON: BM25 + Qdrant Setup

import asyncio
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, 
    Filter, FieldCondition, MatchValue,
    SparseVectorParams, SparseIndexParams
)
import numpy as np
import json

# Sample documents for our knowledge base
DOCUMENTS = [
    {
        "id": 1,
        "title": "Python Async Programming",
        "content": "Asyncio enables concurrent I/O-bound operations in Python. "
                   "The event loop manages coroutines and callbacks efficiently.",
        "source": "tech_docs",
        "category": "programming"
    },
    {
        "id": 2,
        "title": "Machine Learning Basics",
        "content": "Machine learning algorithms learn patterns from training data. "
                   "Neural networks are inspired by biological brain structure.",
        "source": "ml_textbook",
        "category": "ai"
    },
    {
        "id": 3,
        "title": "Vector Databases Explained",
        "content": "Vector databases store high-dimensional embeddings for similarity search. "
                   "They enable semantic search beyond keyword matching.",
        "source": "database_guide",
        "category": "database"
    },
    {
        "id": 4,
        "title": "LLM Token Economics",
        "content": "Large language models process text as tokens. "
                   "API costs are calculated per token in input and output.",
        "source": "ai_guide",
        "category": "ai"
    },
    {
        "id": 5,
        "title": "FastAPI Production Deployment",
        "content": "FastAPI provides high-performance Python web framework. "
                   "It supports async endpoints and automatic OpenAPI documentation.",
        "source": "web_guide",
        "category": "programming"
    },
]

class BM25Retriever:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        
        # Tokenize for BM25
        tokenized = [
            doc["content"].lower().split() 
            for doc in documents
        ]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                results.append({
                    "rank": rank + 1,
                    "score": float(scores[idx]),
                    "document": self.documents[idx]
                })
        
        return results

def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
    id_field: str = "id"
) -> list[dict]:
    """Combine multiple ranked lists using RRF"""
    rrf_scores = {}
    doc_data = {}
    
    for results in result_lists:
        for rank, item in enumerate(results):
            doc_id = item["document"][id_field]
            
            # RRF formula
            rrf_score = 1 / (k + rank + 1)
            
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
                doc_data[doc_id] = item["document"]
            
            rrf_scores[doc_id] += rrf_score
    
    # Sort by RRF score
    sorted_docs = sorted(
        rrf_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return [
        {
            "rrf_score": score,
            "document": doc_data[doc_id]
        }
        for doc_id, score in sorted_docs
    ]

# Test BM25 + basic fusion
bm25 = BM25Retriever(DOCUMENTS)

query = "Python async programming"
results = bm25.search(query, top_k=3)
print(f"BM25 results for '{query}':")
for r in results:
    print(f"  Rank {r['rank']}: {r['document']['title']} (score: {r['score']:.3f})")

# Qdrant setup
async def setup_qdrant():
    client = AsyncQdrantClient(":memory:")  # in-memory for testing
    
    COLLECTION_NAME = "documents"
    
    # Create collection
    await client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,  # sentence-transformers default
            distance=Distance.COSINE
        )
    )
    
    print(f"✅ Created Qdrant collection: {COLLECTION_NAME}")
    
    # Generate embeddings and upsert
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = [doc["content"] for doc in DOCUMENTS]
    embeddings = model.encode(texts).tolist()
    
    points = [
        PointStruct(
            id=doc["id"],
            vector=embedding,
            payload={
                "title": doc["title"],
                "content": doc["content"],
                "source": doc["source"],
                "category": doc["category"]
            }
        )
        for doc, embedding in zip(DOCUMENTS, embeddings)
    ]
    
    await client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    print(f"✅ Inserted {len(points)} documents")
    
    # Test search
    query = "how do language models handle text input?"
    query_embedding = model.encode([query])[0].tolist()
    
    results = await client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=3
    )
    
    print(f"\nVector search results for: '{query}'")
    for r in results:
        print(f"  Score: {r.score:.3f} | {r.payload['title']}")
    
    return client

asyncio.run(setup_qdrant())