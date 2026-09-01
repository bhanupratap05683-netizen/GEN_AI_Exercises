# HANDS-ON: Security + Cost Layer

import asyncio
import time
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# PII Detector
class PIIDetector:
    PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    }
    
    def detect(self, text: str) -> list[dict]:
        found = []
        for pii_type, pattern in self.PATTERNS.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                found.append({
                    "type": pii_type,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        return found
    
    def redact(self, text: str) -> tuple[str, list[dict]]:
        detected = self.detect(text)
        redacted_text = text
        offset = 0
        
        detected_sorted = sorted(detected, key=lambda x: x["start"])
        
        for item in detected_sorted:
            replacement = f"[{item['type'].upper()}_REDACTED]"
            start = item["start"] + offset
            end = item["end"] + offset
            
            redacted_text = redacted_text[:start] + replacement + redacted_text[end:]
            offset += len(replacement) - (item["end"] - item["start"])
        
        return redacted_text, detected

# Token Bucket Rate Limiter
class TokenBucketRateLimiter:
    def __init__(
        self,
        rate: float,  # tokens per second
        capacity: float,  # max tokens in bucket
    ):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            
            # Refill bucket
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    async def wait_and_acquire(self, tokens: int = 1, timeout: float = 60):
        start = time.time()
        while time.time() - start < timeout:
            if await self.acquire(tokens):
                return True
            await asyncio.sleep(0.1)
        raise TimeoutError("Rate limit wait timeout")

# Cost Budget Manager
@dataclass
class UserBudget:
    user_id: str
    monthly_limit_usd: float
    current_usage_usd: float = 0.0
    token_count: int = 0
    reset_date: float = field(default_factory=time.time)
    
    @property
    def remaining(self) -> float:
        return self.monthly_limit_usd - self.current_usage_usd
    
    @property
    def usage_percent(self) -> float:
        return (self.current_usage_usd / self.monthly_limit_usd) * 100

class CostBudgetManager:
    def __init__(self):
        self.budgets: dict[str, UserBudget] = {}
        self.model_costs = {
            "gpt-4o": 0.000015,
            "gpt-4o-mini": 0.0000006,
            "gemini-1.5-flash": 0.00000015,
        }
    
    def create_budget(self, user_id: str, monthly_limit: float):
        self.budgets[user_id] = UserBudget(
            user_id=user_id,
            monthly_limit_usd=monthly_limit
        )
    
    async def check_and_charge(
        self,
        user_id: str,
        model: str,
        tokens: int
    ) -> tuple[bool, str]:
        if user_id not in self.budgets:
            return True, "No budget limit"
        
        budget = self.budgets[user_id]
        cost = (tokens / 1000) * self.model_costs.get(model, 0.001)
        
        # Check budget
        if budget.current_usage_usd + cost > budget.monthly_limit_usd:
            return False, f"Budget exceeded: ${budget.current_usage_usd:.4f}/${budget.monthly_limit_usd}"
        
        # Warn at 80%
        if budget.usage_percent >= 80:
            print(f"⚠️  User {user_id}: {budget.usage_percent:.1f}% of budget used")
        
        # Charge
        budget.current_usage_usd += cost
        budget.token_count += tokens
        
        return True, f"Charged ${cost:.6f}. Remaining: ${budget.remaining:.4f}"

# Tool Permission System
class PermissionLevel(str):
    READ_ONLY = "read_only"
    STANDARD = "standard"
    ADMIN = "admin"

class ToolPermissionGate:
    TOOL_PERMISSIONS = {
        "search_web": PermissionLevel.READ_ONLY,
        "query_database": PermissionLevel.READ_ONLY,
        "write_database": PermissionLevel.STANDARD,
        "send_email": PermissionLevel.STANDARD,
        "delete_records": PermissionLevel.ADMIN,
        "deploy_code": PermissionLevel.ADMIN,
    }
    
    USER_LEVELS = {
        "free": PermissionLevel.READ_ONLY,
        "paid": PermissionLevel.STANDARD,
        "admin": PermissionLevel.ADMIN,
    }
    
    LEVEL_HIERARCHY = {
        PermissionLevel.READ_ONLY: 1,
        PermissionLevel.STANDARD: 2,
        PermissionLevel.ADMIN: 3,
    }
    
    def can_use_tool(self, user_tier: str, tool_name: str) -> tuple[bool, str]:
        user_level = self.USER_LEVELS.get(user_tier, PermissionLevel.READ_ONLY)
        required_level = self.TOOL_PERMISSIONS.get(tool_name, PermissionLevel.ADMIN)
        
        user_rank = self.LEVEL_HIERARCHY.get(user_level, 0)
        required_rank = self.LEVEL_HIERARCHY.get(required_level, 99)
        
        if user_rank >= required_rank:
            return True, "Permitted"
        else:
            return False, f"Tool '{tool_name}' requires {required_level} access (you have {user_level})"

# Test all security layers
async def main():
    # PII detection
    pii = PIIDetector()
    text = "Contact John at john@example.com or 555-123-4567"
    redacted, found = pii.redact(text)
    print(f"Original: {text}")
    print(f"Redacted: {redacted}")
    print(f"Found PII: {[f['type'] for f in found]}")
    
    # Rate limiting
    limiter = TokenBucketRateLimiter(rate=5, capacity=5)
    
    success_count = 0
    for i in range(10):
        success = await limiter.acquire(1)
        if success:
            success_count += 1
    
    print(f"\nRate limiter: {success_count}/10 requests allowed")
    
    # Budget management
    budget_mgr = CostBudgetManager()
    budget_mgr.create_budget("user123", monthly_limit=5.0)
    
    allowed, msg = await budget_mgr.check_and_charge("user123", "gpt-4o", 10000)
    print(f"\nBudget check: {allowed} - {msg}")
    
    # Permission gate
    gate = ToolPermissionGate()
    
    tests = [
        ("free", "search_web"),
        ("free", "send_email"),
        ("paid", "send_email"),
        ("paid", "delete_records"),
        ("admin", "delete_records"),
    ]
    
    print("\nPermission checks:")
    for tier, tool in tests:
        allowed, reason = gate.can_use_tool(tier, tool)
        icon = "✅" if allowed else "❌"
        print(f"  {icon} {tier} → {tool}: {reason}")

asyncio.run(main())