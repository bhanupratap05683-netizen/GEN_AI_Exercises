# Complete Hybrid RAG Engine: hybrid_rag/

# File: hybrid_rag/pipeline.py

import asyncio
import numpy as np
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import openai
import cohere
import re

class HybridRAGEngine:
    def __init__(
        self,
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.collection_name = collection_name
        
        # Models
        self.embedder = SentenceTransformer(embedding_model)
        self.reranker = CrossEncoder(reranker_model)
        
        # Clients
        self.qdrant = AsyncQdrantClient(":memory:")
        self.openai = openai.AsyncOpenAI()
        
        # BM25 components
        self.bm25 = None
        self.documents = []
        
    async def initialize(self):
        await self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )
    
    def _chunk_text(
        self, 
        text: str, 
        chunk_size: int = 400,
        overlap: int = 80
    ) -> list[str]:
        """Recursive chunking strategy"""
        if len(text) <= chunk_size:
            return [text]
        
        separators = ["\n\n", "\n", ". ", " "]
        
        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                chunks = []
                current = ""
                
                for part in parts:
                    if len(current) + len(part) <= chunk_size:
                        current += part + sep
                    else:
                        if current:
                            chunks.append(current.strip())
                        # Overlap
                        overlap_start = max(0, len(current) - overlap)
                        current = current[overlap_start:] + part + sep
                
                if current:
                    chunks.append(current.strip())
                
                return [c for c in chunks if len(c) > 50]
        
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]
    
    async def ingest_documents(
        self, 
        documents: list[dict],
        chunk_documents: bool = True
    ):
        """Ingest documents into vector DB and BM25 index"""
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            content = doc.get("content", "")
            
            if chunk_documents:
                chunks = self._chunk_text(content)
            else:
                chunks = [content]
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": len(all_chunks),
                    "content": chunk,
                    "title": doc.get("title", ""),
                    "source": doc.get("source", ""),
                    "doc_idx": doc_idx,
                    "chunk_idx": chunk_idx
                })
        
        # Store for BM25
        self.documents = all_chunks
        
        # Build BM25 index
        tokenized = [chunk["content"].lower().split() for chunk in all_chunks]
        self.bm25 = BM25Okapi(tokenized)
        
        # Generate embeddings (batch for efficiency)
        contents = [chunk["content"] for chunk in all_chunks]
        embeddings = self.embedder.encode(
            contents, 
            batch_size=32,
            show_progress_bar=True
        )
        
        # Upsert to Qdrant
        points = [
            PointStruct(
                id=chunk["id"],
                vector=embedding.tolist(),
                payload={
                    "content": chunk["content"],
                    "title": chunk["title"],
                    "source": chunk["source"],
                    "doc_idx": chunk["doc_idx"],
                    "chunk_idx": chunk["chunk_idx"]
                }
            )
            for chunk, embedding in zip(all_chunks, embeddings)
        ]
        
        await self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"✅ Ingested {len(documents)} documents → {len(all_chunks)} chunks")
    
    def _bm25_search(self, query: str, top_k: int = 20) -> list[dict]:
        if not self.bm25:
            return []
        
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [
            {
                "rank": rank + 1,
                "score": float(scores[idx]),
                "document": self.documents[idx]
            }
            for rank, idx in enumerate(top_indices)
            if scores[idx] > 0
        ]
    
    async def _vector_search(self, query: str, top_k: int = 20) -> list[dict]:
        query_embedding = self.embedder.encode([query])[0].tolist()
        
        results = await self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        return [
            {
                "rank": rank + 1,
                "score": float(r.score),
                "document": {
                    "id": r.id,
                    **r.payload
                }
            }
            for rank, r in enumerate(results)
        ]
    
    def _reciprocal_rank_fusion(
        self,
        result_lists: list[list[dict]],
        k: int = 60
    ) -> list[dict]:
        rrf_scores = {}
        doc_data = {}
        
        for results in result_lists:
            for rank, item in enumerate(results):
                doc_id = item["document"]["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1/(k + rank + 1)
                doc_data[doc_id] = item["document"]
        
        return [
            {"rrf_score": score, "document": doc_data[doc_id]}
            for doc_id, score in sorted(
                rrf_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]
    
    def _rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5
    ) -> list[dict]:
        if not candidates:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [
            [query, candidate["document"]["content"]]
            for candidate in candidates
        ]
        
        scores = self.reranker.predict(pairs)
        
        # Attach scores and sort
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
        
        return sorted(
            candidates,
            key=lambda x: x["rerank_score"],
            reverse=True
        )[:top_k]
    
    async def retrieve(
        self,
        query: str,
        retrieval_top_k: int = 20,
        final_top_k: int = 5
    ) -> list[dict]:
        """Full hybrid retrieval: BM25 + Vector → RRF → Rerank"""
        
        # Concurrent retrieval
        bm25_results, vector_results = await asyncio.gather(
            asyncio.get_event_loop().run_in_executor(
                None, self._bm25_search, query, retrieval_top_k
            ),
            self._vector_search(query, retrieval_top_k)
        )
        
        print(f"  BM25: {len(bm25_results)} results")
        print(f"  Vector: {len(vector_results)} results")
        
        # Fuse with RRF
        fused = self._reciprocal_rank_fusion([bm25_results, vector_results])
        print(f"  After RRF fusion: {len(fused)} unique results")
        
        # Rerank
        reranked = self._rerank(query, fused[:30], top_k=final_top_k)
        print(f"  After reranking: {len(reranked)} final results")
        
        return reranked
    
    async def query(self, question: str) -> dict:
        """Full RAG: retrieve + generate"""
        
        print(f"\n🔍 Query: {question}")
        
        # Retrieve
        contexts = await self.retrieve(question)
        
        if not contexts:
            return {
                "answer": "I couldn't find relevant information.",
                "sources": [],
                "contexts_used": 0
            }
        
        # Build augmented prompt
        context_text = "\n\n---\n\n".join([
            f"Source: {ctx['document']['title']}\n{ctx['document']['content']}"
            for ctx in contexts
        ])
        
        system = """You are a helpful assistant that answers questions based on provided context.
Rules:
- Only use information from the provided context
- If the context doesn't contain the answer, say so
- Cite your sources by mentioning document titles
- Be concise and accurate"""
        
        user = f"""Context:
{context_text}

Question: {question}

Answer based only on the context above:"""
        
        response = await self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0
        )
        
        return {
            "answer": response.choices[0].message.content,
            "sources": [ctx["document"]["title"] for ctx in contexts],
            "contexts_used": len(contexts),
            "contexts": contexts
        }

# Run the full pipeline
async def main():
    # Initialize engine
    engine = HybridRAGEngine()
    await engine.initialize()
    
    # Sample documents
    documents = [
        {
            "title": "Introduction to Async Python",
            "source": "python_docs",
            "content": """Python's asyncio library enables concurrent programming 
            through coroutines. The event loop processes async tasks efficiently.
            Use async def to define coroutines and await to pause execution.
            This enables handling multiple I/O operations without blocking."""
        },
        {
            "title": "Vector Database Architecture",
            "source": "tech_blog",
            "content": """Vector databases store high-dimensional embeddings.
            They use approximate nearest neighbor algorithms like HNSW.
            Qdrant uses Rust for performance and offers filtering capabilities.
            Collections store vectors with associated metadata payloads."""
        },
        {
            "title": "RAG System Design",
            "source": "ai_handbook",
            "content": """Retrieval Augmented Generation combines search with LLMs.
            First retrieve relevant documents, then pass to language model.
            Hybrid search combining dense and sparse retrieval works best.
            Reranking improves precision by scoring query-document pairs together."""
        },
    ]
    
    await engine.ingest_documents(documents)
    
    # Test queries
    questions = [
        "How does Python handle concurrent operations?",
        "What algorithm do vector databases use for search?",
        "How does RAG improve LLM responses?",
    ]
    
    for question in questions:
        result = await engine.query(question)
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")

asyncio.run(main())