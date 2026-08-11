# HANDS-ON: Complete Async API Client

import asyncio
import openai
from typing import Optional
import time

# Basic async call
async def basic_llm_call(prompt: str, system: str = "") -> str:
    client = openai.AsyncOpenAI()
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )
    
    return response.choices[0].message.content

# Concurrent calls demonstration
async def demonstrate_async_power():
    prompts = [
        "What is 2+2?",
        "Name a color",
        "Name an animal",
        "Name a country",
    ]
    
    # Sequential (slow)
    start = time.time()
    sequential_results = []
    for prompt in prompts:
        result = await basic_llm_call(prompt)
        sequential_results.append(result)
    sequential_time = time.time() - start
    
    # Concurrent (fast)
    start = time.time()
    concurrent_results = await asyncio.gather(
        *[basic_llm_call(p) for p in prompts]
    )
    concurrent_time = time.time() - start
    
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"Concurrent: {concurrent_time:.2f}s")
    print(f"Speedup: {sequential_time/concurrent_time:.1f}x")

# Rate-limited async calls
async def rate_limited_calls(prompts: list, max_concurrent: int = 3):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def call_with_limit(prompt: str) -> str:
        async with semaphore:
            return await basic_llm_call(prompt)
    
    results = await asyncio.gather(
        *[call_with_limit(p) for p in prompts]
    )
    return results

# Streaming response
async def stream_response(prompt: str):
    client = openai.AsyncOpenAI()
    
    print("Streaming: ", end="", flush=True)
    
    async with client.chat.completions.stream(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)
    print()  # newline at end

asyncio.run(demonstrate_async_power())