# HANDS-ON: Redis Memory + Multi-Agent

import asyncio
import redis.asyncio as aioredis
import json
import time
from typing import Optional
import openai

class AgentMemory:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = aioredis.from_url(redis_url)
    
    async def save_conversation(
        self, 
        session_id: str, 
        role: str, 
        content: str,
        ttl: int = 3600
    ):
        key = f"session:{session_id}:messages"
        message = json.dumps({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        await self.redis.rpush(key, message)
        await self.redis.expire(key, ttl)
    
    async def get_conversation(self, session_id: str) -> list[dict]:
        key = f"session:{session_id}:messages"
        messages = await self.redis.lrange(key, 0, -1)
        return [json.loads(m) for m in messages]
    
    async def save_user_profile(
        self, 
        user_id: str, 
        profile_data: dict
    ):
        key = f"user:{user_id}:profile"
        await self.redis.hset(key, mapping={
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in profile_data.items()
        })
    
    async def get_user_profile(self, user_id: str) -> dict:
        key = f"user:{user_id}:profile"
        data = await self.redis.hgetall(key)
        return {
            k.decode(): v.decode() 
            for k, v in data.items()
        }
    
    async def summarize_and_compress(
        self, 
        session_id: str,
        keep_recent: int = 5
    ):
        """Compress old messages into summary"""
        messages = await self.get_conversation(session_id)
        
        if len(messages) <= keep_recent:
            return
        
        old_messages = messages[:-keep_recent]
        recent_messages = messages[-keep_recent:]
        
        client = openai.AsyncOpenAI()
        
        conversation_text = "\n".join([
            f"{m['role']}: {m['content']}"
            for m in old_messages
        ])
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize this conversation in 3 sentences:\n\n{conversation_text}"
                }
            ]
        )
        
        summary = response.choices[0].message.content
        
        # Replace old messages with summary
        key = f"session:{session_id}:messages"
        await self.redis.delete(key)
        
        # Add summary as system message
        await self.save_conversation(
            session_id, "system",
            f"[CONVERSATION SUMMARY]: {summary}"
        )
        
        # Add recent messages back
        for msg in recent_messages:
            await self.save_conversation(
                session_id, msg["role"], msg["content"]
            )
        
        print(f"✅ Compressed {len(old_messages)} messages into summary")

class SpecialistAgent:
    def __init__(self, name: str, specialty: str, tools: list):
        self.name = name
        self.specialty = specialty
        self.tools = tools
        self.client = openai.AsyncOpenAI()
    
    async def handle(self, task: str, context: dict = None) -> dict:
        """Handle a task within this agent's specialty"""
        system = f"""You are {self.name}, specialized in {self.specialty}.
Only handle tasks related to your specialty.
If a task is outside your domain, say so clearly.
Be concise and accurate."""
        
        context_str = f"\nContext: {json.dumps(context)}" if context else ""
        
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Task: {task}{context_str}"}
            ]
        )
        
        return {
            "agent": self.name,
            "specialty": self.specialty,
            "response": response.choices[0].message.content
        }

class SupervisorAgent:
    def __init__(self, specialists: list[SpecialistAgent]):
        self.specialists = {s.name: s for s in specialists}
        self.client = openai.AsyncOpenAI()
    
    async def route(self, task: str) -> str:
        """Decide which specialist(s) to use"""
        specialist_list = "\n".join([
            f"- {s.name}: {s.specialty}"
            for s in self.specialists.values()
        ])
        
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a routing supervisor.
Route tasks to the best specialist.
Specialists:
{specialist_list}

Return JSON: {{"specialist": "name", "reasoning": "why"}}"""
                },
                {"role": "user", "content": f"Route this task: {task}"}
            ],
            response_format={"type": "json_object"}
        )
        
        routing = json.loads(response.choices[0].message.content)
        return routing["specialist"]
    
    async def execute(self, task: str) -> dict:
        # Route to best specialist
        specialist_name = await self.route(task)
        
        if specialist_name not in self.specialists:
            return {"error": f"Unknown specialist: {specialist_name}"}
        
        specialist = self.specialists[specialist_name]
        result = await specialist.handle(task)
        
        return {
            "task": task,
            "routed_to": specialist_name,
            "result": result
        }

# Test multi-agent system
async def main():
    specialists = [
        SpecialistAgent("BillingAgent", "invoices, payments, pricing, subscriptions", []),
        SpecialistAgent("TechSupportAgent", "technical issues, bugs, API errors, debugging", []),
        SpecialistAgent("SalesAgent", "product information, demos, quotes, contracts", []),
    ]
    
    supervisor = SupervisorAgent(specialists)
    
    tasks = [
        "My invoice shows wrong amount for last month",
        "I'm getting a 401 error when calling your API",
        "Can you tell me about your enterprise pricing?",
    ]
    
    for task in tasks:
        result = await supervisor.execute(task)
        print(f"\nTask: {task}")
        print(f"Routed to: {result['routed_to']}")
        print(f"Response: {result['result']['response'][:150]}...")

asyncio.run(main())