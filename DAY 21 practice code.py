# HANDS-ON: Intelligent Model Router

import asyncio
import litellm
from enum import Enum
from pydantic import BaseModel
import openai
import re

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

class ModelConfig:
    ROUTING_TABLE = {
        TaskComplexity.SIMPLE: {
            "primary": "gemini/gemini-1.5-flash",
            "fallback": "gpt-4o-mini",
            "max_tokens": 256,
            "cost_per_1k_tokens": 0.00015
        },
        TaskComplexity.MEDIUM: {
            "primary": "gpt-4o-mini",
            "fallback": "gemini/gemini-1.5-pro",
            "max_tokens": 1024,
            "cost_per_1k_tokens": 0.0006
        },
        TaskComplexity.COMPLEX: {
            "primary": "gpt-4o",
            "fallback": "gemini/gemini-1.5-pro",
            "max_tokens": 4096,
            "cost_per_1k_tokens": 0.015
        }
    }

class IntelligentRouter:
    def __init__(self):
        self.client = openai.AsyncOpenAI()
        self.total_cost = 0.0
        self.call_count = 0
    
    def classify_complexity(self, prompt: str) -> TaskComplexity:
        # Heuristic classification
        word_count = len(prompt.split())
        
        complex_indicators = [
            "analyze", "compare", "evaluate", "explain why",
            "design", "architect", "implement", "debug",
            "multiple steps", "comprehensive", "detailed analysis"
        ]
        
        simple_indicators = [
            "what is", "define", "list", "name",
            "when was", "who is", "yes or no", "true or false"
        ]
        
        prompt_lower = prompt.lower()
        
        complex_score = sum(1 for ind in complex_indicators if ind in prompt_lower)
        simple_score = sum(1 for ind in simple_indicators if ind in prompt_lower)
        
        if word_count > 200 or complex_score >= 2:
            return TaskComplexity.COMPLEX
        elif word_count < 30 or simple_score >= 2:
            return TaskComplexity.SIMPLE
        else:
            return TaskComplexity.MEDIUM
    
    async def complete(
        self, 
        prompt: str,
        system: str = "",
        force_model: str = None
    ) -> dict:
        complexity = self.classify_complexity(prompt)
        config = ModelConfig.ROUTING_TABLE[complexity]
        
        model = force_model or config["primary"]
        
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": system} if system else None,
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config["max_tokens"]
            )
            
            tokens_used = response.usage.total_tokens
            cost = (tokens_used / 1000) * config["cost_per_1k_tokens"]
            
            self.total_cost += cost
            self.call_count += 1
            
            return {
                "response": response.choices[0].message.content,
                "model_used": model,
                "complexity": complexity.value,
                "tokens": tokens_used,
                "estimated_cost": f"${cost:.6f}",
                "total_cost_session": f"${self.total_cost:.4f}"
            }
            
        except Exception as e:
            # Fallback to alternative model
            print(f"⚠️  {model} failed: {e}. Falling back...")
            fallback_model = config["fallback"]
            
            response = await litellm.acompletion(
                model=fallback_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                "response": response.choices[0].message.content,
                "model_used": fallback_model,
                "complexity": complexity.value,
                "fallback": True
            }

# Test the router
async def main():
    router = IntelligentRouter()
    
    test_prompts = [
        "What is Python?",  # SIMPLE
        "Explain the difference between REST and GraphQL",  # MEDIUM
        "Design a microservices architecture for a high-traffic e-commerce platform with detailed component breakdown",  # COMPLEX
    ]
    
    for prompt in test_prompts:
        result = await router.complete(prompt)
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"Complexity: {result['complexity']}")
        print(f"Model: {result['model_used']}")
        print(f"Cost: {result.get('estimated_cost', 'N/A')}")

asyncio.run(main())