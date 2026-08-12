# HANDS-ON: Multi-Provider LLM Client

import asyncio
from abc import ABC, abstractmethod
import openai
import google.generativeai as genai
from pydantic import BaseModel
import json
from typing import Any

class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, system: str, user: str, 
                      temperature: float = 0) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = openai.AsyncOpenAI()
        self.model = model
    
    async def complete(self, system: str, user: str,
                      temperature: float = 0) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

class GeminiProvider(LLMProvider):
    def __init__(self, model: str = "gemini-1.5-flash"):
        genai.configure(api_key="YOUR_GEMINI_KEY")
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )
    
    async def complete(self, system: str, user: str,
                      temperature: float = 0) -> str:
        loop = asyncio.get_event_loop()
        
        prompt = f"{system}\n\n{user}"
        
        response = await loop.run_in_executor(
            None,
            lambda: self.model.generate_content(prompt)
        )
        return response.text

class FallbackProvider(LLMProvider):
    def __init__(self, providers: list[LLMProvider]):
        self.providers = providers
    
    async def complete(self, system: str, user: str,
                      temperature: float = 0) -> str:
        for i, provider in enumerate(self.providers):
            try:
                return await provider.complete(system, user, temperature)
            except Exception as e:
                print(f"Provider {i} failed: {e}, trying next...")
        raise Exception("All providers failed")

# Usage
provider = FallbackProvider([
    OpenAIProvider("gpt-4o-mini"),
    GeminiProvider("gemini-1.5-flash")
])