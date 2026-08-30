# HANDS-ON: FastAPI Streaming Agent Service

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json
import openai
import time
from typing import AsyncGenerator

app = FastAPI(title="AI Agent API", version="1.0.0")

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    stream: bool = True

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.92):
        self.cache: list[dict] = []  # In prod: Redis Vector
        self.threshold = similarity_threshold
        
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _similarity(self, a: list, b: list) -> float:
        import numpy as np
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    async def get(self, query: str) -> tuple[bool, str | None]:
        embedding = self.embedder.encode([query])[0].tolist()
        
        for entry in self.cache:
            similarity = self._similarity(embedding, entry["embedding"])
            if similarity >= self.threshold:
                print(f"🎯 Cache hit! Similarity: {similarity:.3f}")
                return True, entry["response"]
        
        return False, None
    
    async def set(self, query: str, response: str):
        embedding = self.embedder.encode([query])[0].tolist()
        self.cache.append({
            "query": query,
            "embedding": embedding,
            "response": response,
            "cached_at": time.time()
        })

semantic_cache = SemanticCache()

async def stream_llm_response(message: str) -> AsyncGenerator[str, None]:
    client = openai.AsyncOpenAI()
    
    # Send a start event
    yield f"data: {json.dumps({'type': 'start', 'message': 'Generating...'})}\n\n"
    
    full_response = ""
    
    async with client.chat.completions.stream(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ]
    ) as stream:
        async for text in stream.text_stream:
            full_response += text
            
            yield f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"
    
    # Cache the complete response
    await semantic_cache.set(message, full_response)
    
    # Send completion event
    yield f"data: {json.dumps({'type': 'done', 'message': 'Complete'})}\n\n"

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    # Check semantic cache first
    cache_hit, cached_response = await semantic_cache.get(request.message)
    
    if cache_hit:
        async def stream_cached():
            yield f"data: {json.dumps({'type': 'cache_hit', 'content': cached_response})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        return StreamingResponse(
            stream_cached(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Cache": "HIT"
            }
        )
    
    return StreamingResponse(
        stream_llm_response(request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Cache": "MISS"
        }
    )

@app.post("/chat")
async def chat(request: ChatRequest):
    cache_hit, cached_response = await semantic_cache.get(request.message)
    
    if cache_hit:
        return {
            "response": cached_response,
            "cached": True,
            "session_id": request.session_id
        }
    
    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": request.message}]
    )
    
    answer = response.choices[0].message.content
    await semantic_cache.set(request.message, answer)
    
    return {
        "response": answer,
        "cached": False,
        "session_id": request.session_id,
        "usage": {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "cache_size": len(semantic_cache.cache)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)